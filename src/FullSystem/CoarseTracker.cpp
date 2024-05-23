/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {


    template<int b, typename T>
    T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
        const int padT = 1 + ((1 << b) / sizeof(T));
        T *ptr = new T[size + padT];
        rawPtrVec.push_back(ptr);
        T *alignedPtr = (T *) ((((uintptr_t) (ptr + padT)) >> b) << b);
        return alignedPtr;
    }


    CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0,0) // JZ modification: affLight
    {
        // make coarse tracking templates.
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            int wl = ww >> lvl;
            int hl = hh >> lvl;

            idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            weightSums[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            weightSums_bak[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

            pc_u[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_v[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_color[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            // JZ modification: previous frame of the last keyframe (lastRef)
            pc_color_prev[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

        }

        // warped buffers
        buf_warped_idepth = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_u = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_v = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_dx = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_dy = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_residual = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_weight = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_refColor = allocAligned<4, float>(ww * hh, ptrToDelete);
        // JZ modification: previous frame (intensity residual of the lastRef, not previous intensity itself)
        buf_warped_prevColorRes = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_refColorPrev = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_newColorPrev = allocAligned<4, float>(ww * hh, ptrToDelete);


        newFrame = 0;
        lastRef = 0;
        debugPlot = debugPrint = true;
        w[0] = h[0] = 0;
        refFrameID = -1;
    }

    CoarseTracker::~CoarseTracker() {
        for (float *ptr: ptrToDelete)
            delete[] ptr;
        ptrToDelete.clear();
    }

    void CoarseTracker::makeK(CalibHessian *HCalib) {
        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }


    void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians) {
        // make coarse tracking templates for latstRef.
        memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
        memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

        // project all PointHessians into the newest keyframe (lastRef). Calculate their idpeth and associated weights.
        for (FrameHessian *fh: frameHessians) {
            for (PointHessian *ph: fh->pointHessians) {
                // only if the PointHessian has a good residual last time
                if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) {
                    PointFrameResidual *r = ph->lastResiduals[0].first;
                    assert(r->efResidual->isActive() && r->target == lastRef);
                    int u = r->centerProjectedTo[0] + 0.5f;
                    int v = r->centerProjectedTo[1] + 0.5f;
                    float new_idepth = r->centerProjectedTo[2];
                    float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

                    idepth[0][u + w[0] * v] += new_idepth * weight;
                    weightSums[0][u + w[0] * v] += weight;
                }
            }
        }


        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            float *idepth_l = idepth[lvl];
            float *weightSums_l = weightSums[lvl];

            float *idepth_lm = idepth[lvlm1];
            float *weightSums_lm = weightSums[lvlm1];

            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++) {
                    int bidx = 2 * x + 2 * y * wlm1;
                    idepth_l[x + y * wl] = idepth_lm[bidx] +
                                           idepth_lm[bidx + 1] +
                                           idepth_lm[bidx + wlm1] +
                                           idepth_lm[bidx + wlm1 + 1];

                    weightSums_l[x + y * wl] = weightSums_lm[bidx] +
                                               weightSums_lm[bidx + 1] +
                                               weightSums_lm[bidx + wlm1] +
                                               weightSums_lm[bidx + wlm1 + 1];
                }
        }


        // dilate idepth by 1.
        for (int lvl = 0; lvl < 2; lvl++) {
            int numIts = 1;


            for (int it = 0; it < numIts; it++) {
                int wh = w[lvl] * h[lvl] - w[lvl];
                int wl = w[lvl];
                float *weightSumsl = weightSums[lvl];
                float *weightSumsl_bak = weightSums_bak[lvl];
                memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
                float *idepthl = idepth[lvl];    // dotnt need to make a temp copy of depth, since I only
                // read values with weightSumsl>0, and write ones with weightSumsl<=0.
                for (int i = w[lvl]; i < wh; i++) {
                    if (weightSumsl_bak[i] <= 0) {
                        float sum = 0, num = 0, numn = 0;
                        if (weightSumsl_bak[i + 1 + wl] > 0) {
                            sum += idepthl[i + 1 + wl];
                            num += weightSumsl_bak[i + 1 + wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i - 1 - wl] > 0) {
                            sum += idepthl[i - 1 - wl];
                            num += weightSumsl_bak[i - 1 - wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i + wl - 1] > 0) {
                            sum += idepthl[i + wl - 1];
                            num += weightSumsl_bak[i + wl - 1];
                            numn++;
                        }
                        if (weightSumsl_bak[i - wl + 1] > 0) {
                            sum += idepthl[i - wl + 1];
                            num += weightSumsl_bak[i - wl + 1];
                            numn++;
                        }
                        if (numn > 0) {
                            idepthl[i] = sum / numn;
                            weightSumsl[i] = num / numn;
                        }
                    }
                }
            }
        }


        // dilate idepth by 1 (2 on lower levels).
        for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            float *weightSumsl = weightSums[lvl];
            float *weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
            float *idepthl = idepth[lvl];    // dotnt need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++) {
                if (weightSumsl_bak[i] <= 0) {
                    float sum = 0, num = 0, numn = 0;
                    if (weightSumsl_bak[i + 1] > 0) {
                        sum += idepthl[i + 1];
                        num += weightSumsl_bak[i + 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i - 1] > 0) {
                        sum += idepthl[i - 1];
                        num += weightSumsl_bak[i - 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i + wl] > 0) {
                        sum += idepthl[i + wl];
                        num += weightSumsl_bak[i + wl];
                        numn++;
                    }
                    if (weightSumsl_bak[i - wl] > 0) {
                        sum += idepthl[i - wl];
                        num += weightSumsl_bak[i - wl];
                        numn++;
                    }
                    if (numn > 0) {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }


        // normalize idepths and weights.
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            float *weightSumsl = weightSums[lvl];
            float *idepthl = idepth[lvl];
            Eigen::Vector3f *dIRefl = lastRef->dIp[lvl];    // lastRef is a keyframe
            // JZ modification: previous frame of lastRef
            Eigen::Vector3f *dIRefPrevl = lastRefPrev->dIp[lvl];

            int wl = w[lvl], hl = h[lvl];

            int lpc_n = 0;
            float *lpc_u = pc_u[lvl];
            float *lpc_v = pc_v[lvl];
            float *lpc_idepth = pc_idepth[lvl];
            float *lpc_color = pc_color[lvl];
            // JZ modification: previous frame of the last keyframe
            float *lpc_color_prev = pc_color_prev[lvl];


            for (int y = 2; y < hl - 2; y++)
                for (int x = 2; x < wl - 2; x++) {
                    int i = x + y * wl;
                    if (weightSumsl[i] > 0)      // weightSumsl has a value only if the projection of the PointHessians on the lastRef exists.
                    {
                        idepthl[i] /= weightSumsl[i];
                        lpc_u[lpc_n] = x;
                        lpc_v[lpc_n] = y;
                        lpc_idepth[lpc_n] = idepthl[i];
                        lpc_color[lpc_n] = dIRefl[i][0];        // lpc_color and pc_color are both assigned and ready for use (same pointer)
                        // JZ modification: previous frame of the last keyframe
                        lpc_color_prev[lpc_n] = dIRefPrevl[i][0];


                        if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0)) {
                            idepthl[i] = -1;
                            continue;    // just skip if something is wrong.
                        }
                        lpc_n++;
                    } else
                        idepthl[i] = -1;

                    weightSumsl[i] = 1;
                }

            pc_n[lvl] = lpc_n;
        }

    }


    void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew,
                                  AffLight aff_g2l) // AffLight aff_g2l)    // JZ modification: affLight
    {
        acc.initialize();

        __m128 fxl = _mm_set1_ps(fx[lvl]);
        __m128 fyl = _mm_set1_ps(fy[lvl]);
        // JZ modification: affLight
        //__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
        //__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));
        Vec4 affThermal = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRefPrev->ab_exposure, prevFrame->ab_exposure,
                                                      lastRef_aff_g2l, aff_g2l, IR_Data);

        __m128 one = _mm_set1_ps(1);
        __m128 minusOne = _mm_set1_ps(-1);
        __m128 zero = _mm_set1_ps(0);
        __m128 resRatio = _mm_set1_ps(0.00);

        int n = buf_warped_n;
        assert(n % 4 == 0);
        for (int i = 0; i < n; i += 4) {
            __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl);
            __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl);
            __m128 u = _mm_load_ps(buf_warped_u + i);
            __m128 v = _mm_load_ps(buf_warped_v + i);
            __m128 id = _mm_load_ps(buf_warped_idepth + i);


            // JZ modification: affLight (gradient)
            __m128 J6, J7;
            if (!IR_Data) {
                __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
                __m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRefPrev->ab_exposure, prevFrame->ab_exposure, lastRef_aff_g2l, aff_g2l, IR_Data)[0]));
                J6 = _mm_mul_ps(a,_mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor+i))),
                J7 = minusOne;
            }
            else {
                if (ir_correction_model == 0) {
                    // ideal model
                    float A1 = affThermal[2];
                    float B1 = affThermal[3];
                    float oneOverA1_float = 1 / A1;
                    float B1OverA1_float = B1 / A1;
                    float dBdb2_float = (prevFrame->ab_exposure - setting_frameTime) * affThermal[1];
                    __m128 oneOverA1 = _mm_set1_ps(oneOverA1_float);
                    __m128 B1OverA1 = _mm_set1_ps(B1OverA1_float);
                    __m128 drdA = _mm_add_ps(_mm_mul_ps(minusOne, _mm_load_ps(buf_warped_refColor+i)), _mm_mul_ps(_mm_set1_ps(B1), _mm_load_ps(buf_warped_refColorPrev+i)));
                    __m128 dAda2 = _mm_mul_ps(_mm_mul_ps(oneOverA1, _mm_sub_ps(_mm_set1_ps((float)(affThermal[0])), one)), _mm_set1_ps((float)(-newFrame->ab_exposure)));
                    __m128 drdB = _mm_mul_ps(minusOne, _mm_load_ps(buf_warped_refColorPrev+i));//_mm_load_ps(buf_warped_newColorPrev+i); //
                    __m128 dBdb2 = _mm_mul_ps(_mm_set1_ps((float)(affThermal[1])), _mm_set1_ps((float)(prevFrame->ab_exposure - setting_frameTime)));
                    J6 = _mm_mul_ps(drdA, dAda2);
                    J7 = _mm_mul_ps(drdB, dBdb2);
                }
                else if (ir_correction_model == 1) {
                    // exponential non-linear model
                    J6 = minusOne;
                    __m128 dBdb2 = _mm_set1_ps((float) affThermal[1] * (setting_frameTime - prevFrame->ab_exposure));
                    J7 = _mm_mul_ps(_mm_mul_ps(minusOne, _mm_load_ps(buf_warped_refColorPrev + i)), dBdb2);
                }
                else {
                    // linear model
                    J6 = minusOne;
                    J7 = _mm_mul_ps(minusOne, _mm_load_ps(buf_warped_refColorPrev+i));
                }
            }


            acc.updateSSE_eighted(
                    _mm_mul_ps(id, dx),
                    _mm_mul_ps(id, dy),
                    _mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx), _mm_mul_ps(v, dy)))),
                    _mm_sub_ps(zero, _mm_add_ps(
                            _mm_mul_ps(_mm_mul_ps(u, v), dx),
                            _mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))),
                    _mm_add_ps(
                            _mm_mul_ps(_mm_mul_ps(u, v), dy),
                            _mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),
                    _mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),
                    J6,
                    J7,
                    _mm_load_ps(buf_warped_residual + i),
                    _mm_load_ps(buf_warped_weight + i));
        }

        acc.finish();
        H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
        b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

        H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
        H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
        H_out.block<8, 1>(0, 6) *= SCALE_A;
        H_out.block<8, 1>(0, 7) *= SCALE_B;
        H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
        H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
        H_out.block<1, 8>(6, 0) *= SCALE_A;
        H_out.block<1, 8>(7, 0) *= SCALE_B;
        b_out.segment<3>(0) *= SCALE_XI_ROT;
        b_out.segment<3>(3) *= SCALE_XI_TRANS;
        b_out.segment<1>(6) *= SCALE_A;
        b_out.segment<1>(7) *= SCALE_B;
    }


// JZ modification: affLight
    Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH) {
        float E = 0;
        int numTermsInE = 0;
        int numTermsInWarped = 0;
        int numSaturated = 0;

        int wl = w[lvl];
        int hl = h[lvl];
        Eigen::Vector3f *dINewl = newFrame->dIp[lvl];
        // JZ modification: previous frame
        Eigen::Vector3f *dINewlPrev = prevFrame->dIp[lvl];
        float fxl = fx[lvl];
        float fyl = fy[lvl];
        float cxl = cx[lvl];
        float cyl = cy[lvl];


        Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
        Vec3f t = (refToNew.translation()).cast<float>();
        // JZ modification: affLight
        //Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();
        Vec4f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRefPrev->ab_exposure, prevFrame->ab_exposure,
                                                  lastRef_aff_g2l, aff_g2l, IR_Data).cast<float>();


        float sumSquaredShiftT = 0;
        float sumSquaredShiftRT = 0;
        float sumSquaredShiftNum = 0;

        float maxEnergy = 2 * setting_huberTH * cutoffTH - setting_huberTH * setting_huberTH;    // energy for r=setting_coarseCutoffTH.


        MinimalImageB3 *resImage = 0;
        if (debugPlot) {
            resImage = new MinimalImageB3(wl, hl);
            resImage->setConst(Vec3b(255, 255, 255));
        }

        int nl = pc_n[lvl];
        float *lpc_u = pc_u[lvl];
        float *lpc_v = pc_v[lvl];
        float *lpc_idepth = pc_idepth[lvl];
        float *lpc_color = pc_color[lvl];
        // JZ modification: previous frame of the last keyframe
        float *lpc_color_prev = pc_color_prev[lvl];


        for (int i = 0; i < nl; i++) {
            float id = lpc_idepth[i];
            float x = lpc_u[i];
            float y = lpc_v[i];

            Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;
            float new_idepth = id / pt[2];

            if (lvl == 0 && i % 32 == 0) {
                // translation only (positive)
                Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
                float uT = ptT[0] / ptT[2];
                float vT = ptT[1] / ptT[2];
                float KuT = fxl * uT + cxl;
                float KvT = fyl * vT + cyl;

                // translation only (negative)
                Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
                float uT2 = ptT2[0] / ptT2[2];
                float vT2 = ptT2[1] / ptT2[2];
                float KuT2 = fxl * uT2 + cxl;
                float KvT2 = fyl * vT2 + cyl;

                //translation and rotation (negative)
                Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
                float u3 = pt3[0] / pt3[2];
                float v3 = pt3[1] / pt3[2];
                float Ku3 = fxl * u3 + cxl;
                float Kv3 = fyl * v3 + cyl;

                //translation and rotation (positive)
                //already have it.

                sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
                sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
                sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
                sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
                sumSquaredShiftNum += 2;
            }

            if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0)) continue;


            float refColor = lpc_color[i];
            // JZ modification: previous frame of the last keyframe
            float refColorPrev = lpc_color_prev[i];
            Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
            Vec3f hitColorPrev = getInterpolatedElement33(dINewlPrev, Ku, Kv, wl);
            if (!std::isfinite((float) hitColor[0]) || !std::isfinite((float) hitColorPrev[0]) ||
                !std::isfinite(refColorPrev) || !std::isfinite(refColor))
                continue;
            // JZ modification: affLight
            // JZ modification: previous frame (res)
            float residual;
            if (!IR_Data) {
                residual = hitColor[0] - (float) (affLL[0] * refColor + affLL[1]);
            }
            else {
                if (ir_correction_model == 0) {
                    // ideal model
                    float prevValue = (refColorPrev + hitColorPrev[0]) / 2;
                    residual = hitColor[0] - (affLL[0] / affLL[2]) * refColor + (affLL[0] / affLL[2]) * affLL[3] * refColorPrev - affLL[1] * refColorPrev;//hitColorPrev[0];
                }
                else if (ir_correction_model == 1){
                    // non-linear exponential model
                    residual = hitColor[0] - refColor - (affLL[0] - affLL[2]) - (affLL[1] - affLL[3]) * refColorPrev;
                }
                else {
                    float hitColorPrevCoolTime = setting_frameTime - prevFrame->ab_exposure;
                    float refColorPrevCoolTime = setting_frameTime - lastRefPrev->ab_exposure;
                    float hitColorPrevRes = hitColorPrev[0] * pow(affLL[1], hitColorPrevCoolTime);
                    float refColorPrevRes = refColorPrev * pow(affLL[1], refColorPrevCoolTime);
                }
            }
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


            if (fabs(residual) > cutoffTH) {
                if (debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));
                E += maxEnergy;
                numTermsInE++;
                numSaturated++;
            } else {
                if (debugPlot)
                    resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual + 128, residual + 128, residual + 128));

                E += hw * residual * residual * (2 - hw);
                numTermsInE++;

                buf_warped_idepth[numTermsInWarped] = new_idepth;
                buf_warped_u[numTermsInWarped] = u;
                buf_warped_v[numTermsInWarped] = v;
                buf_warped_dx[numTermsInWarped] = hitColor[1];
                buf_warped_dy[numTermsInWarped] = hitColor[2];
                buf_warped_residual[numTermsInWarped] = residual;
                buf_warped_weight[numTermsInWarped] = hw;
                buf_warped_refColor[numTermsInWarped] = lpc_color[i];   // refColor
                // JZ modification: previous frame (intensity residual of the lastRef, not previous intensity itself)
                buf_warped_prevColorRes[numTermsInWarped] = 0.06 * lpc_color_prev[i] * exp(lpc_color_prev[i] / 255.0);
                buf_warped_refColorPrev[numTermsInWarped] = lpc_color_prev[i];
                buf_warped_newColorPrev[numTermsInWarped] = hitColorPrev[0];
                numTermsInWarped++;
            }
        }

        while (numTermsInWarped % 4 != 0) {
            buf_warped_idepth[numTermsInWarped] = 0;
            buf_warped_u[numTermsInWarped] = 0;
            buf_warped_v[numTermsInWarped] = 0;
            buf_warped_dx[numTermsInWarped] = 0;
            buf_warped_dy[numTermsInWarped] = 0;
            buf_warped_residual[numTermsInWarped] = 0;
            buf_warped_weight[numTermsInWarped] = 0;
            buf_warped_refColor[numTermsInWarped] = 0;
            // JZ modification: previous frame (intensity residual of the lastRef, not previous intensity itself)
            buf_warped_prevColorRes[numTermsInWarped] = 0;
            buf_warped_refColorPrev[numTermsInWarped] = 0;
            buf_warped_newColorPrev[numTermsInWarped] = 0;
            numTermsInWarped++;
        }
        buf_warped_n = numTermsInWarped;


        if (debugPlot) {
            IOWrap::displayImage("RES", resImage, false);
            IOWrap::waitKey(0);
            delete resImage;
        }

        Vec6 rs;
        rs[0] = E;
        rs[1] = numTermsInE;
        rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
        rs[3] = 0;
        rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
        rs[5] = numSaturated / (float) numTermsInE;

        return rs;
    }


// JZ modification: previous frame
    void CoarseTracker::setCoarseTrackingRef(
            std::vector<FrameHessian *> frameHessians, FrameHessian *lastKfPrev) {
        assert(!frameHessians.empty());
        lastRef = frameHessians.back();     // last keyframe in the active frame vector
        // JZ modification: previous frame
        lastRefPrev = lastKfPrev;
        makeCoarseDepthL0(frameHessians);


        refFrameID = lastRef->shell->id;
        // JZ modification: affLight
        lastRef_aff_g2l = lastRef->aff_g2l();

        firstCoarseRMSE = -1;

    }

// JZ modification: previous frame
    bool CoarseTracker::trackNewestCoarse(
            FrameHessian *previousFrameHessian,
            FrameHessian *newFrameHessian,
            SE3 &lastToNew_out, AffLight &aff_g2l_out, //AffLight &aff_g2l_out,  // JZ modification: affLight
            int coarsestLvl,
            Vec5 minResForAbort,
            IOWrap::Output3DWrapper *wrap) {
        debugPlot = setting_render_displayCoarseTrackingFull;
        debugPrint = false;

        assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

        lastResiduals.setConstant(NAN);
        lastFlowIndicators.setConstant(1000);


        newFrame = newFrameHessian;
        // JZ modification: previous frame
        prevFrame = previousFrameHessian;
        int maxIterations[] = {10, 20, 50, 50, 50};
        float lambdaExtrapolationLimit = 0.001;

        SE3 refToNew_current = lastToNew_out;
        // JZ modification: affLight
        //AffLight aff_g2l_current = aff_g2l_out;
        AffLight aff_g2l_current = aff_g2l_out;

        bool haveRepeated = false;


        for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
            Mat88 H;
            Vec8 b;
            float levelCutoffRepeat = 1;
            // JZ modification: affLight
            Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);
            while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
                levelCutoffRepeat *= 2;
                // JZ modification: affLight
                // JZ modification: previous frame (modifications are inside the calcRes())
                resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);

                if (!setting_debugout_runquiet)
                    printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH * levelCutoffRepeat,
                           resOld[5]);
            }

            // JZ modification: affLight
            calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

            float lambda = 0.01;

            if (debugPrint) {
                // JZ modification: affLight
                //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
                Vec4f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRefPrev->ab_exposure, prevFrame->ab_exposure,
                                                           lastRef_aff_g2l, aff_g2l_current, IR_Data).cast<float>();
                printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
                       lvl, -1, lambda, 1.0f,
                       "INITIA",
                       0.0f,
                       resOld[0] / resOld[1],
                       0, (int) resOld[1],
                       0.0f);
                std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose()
                          << " (rel " << relAff.transpose() << ")\n";
            }


            for (int iteration = 0; iteration < maxIterations[lvl]; iteration++) {
                Mat88 Hl = H;
                for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);
                Vec8 inc = Hl.ldlt().solve(-b);

                if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0)    // fix a, b
                {
                    inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
                    inc.tail<2>().setZero();
                }
                if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)    // fix b
                {
                    inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
                    inc.tail<1>().setZero();
                }
                if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))    // fix a
                {
                    Mat88 HlStitch = Hl;
                    Vec8 bStitch = b;
                    HlStitch.col(6) = HlStitch.col(7);
                    HlStitch.row(6) = HlStitch.row(7);
                    bStitch[6] = bStitch[7];
                    Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
                    inc.setZero();
                    inc.head<6>() = incStitch.head<6>();
                    inc[6] = 0;
                    inc[7] = incStitch[6];
                }


                float extrapFac = 1;
                if (lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
                inc *= extrapFac;

                Vec8 incScaled = inc;
                incScaled.segment<3>(0) *= SCALE_XI_ROT;
                incScaled.segment<3>(3) *= SCALE_XI_TRANS;
                incScaled.segment<1>(6) *= SCALE_A;
                incScaled.segment<1>(7) *= SCALE_B;

                if (!std::isfinite(incScaled.sum())) incScaled.setZero();

                SE3 refToNew_new = SE3::exp((Vec6) (incScaled.head<6>())) * refToNew_current;
                // JZ modification: affLight
                //AffLight aff_g2l_new = aff_g2l_current;
                //aff_g2l_new.a += incScaled[6];
                //aff_g2l_new.b += incScaled[7];
                AffLight aff_g2l_new = aff_g2l_current;
                aff_g2l_new.a += incScaled[6];
                aff_g2l_new.b += incScaled[7];

                // JZ modification: previous frame (modifications are inside the calcRes())
                Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);

                bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

                if (debugPrint) {
                    // JZ modification: affLight
                    //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
                    Vec4f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRefPrev->ab_exposure, prevFrame->ab_exposure,
                                                               lastRef_aff_g2l, aff_g2l_new, IR_Data).cast<float>();
                    printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
                           lvl, iteration, lambda,
                           extrapFac,
                           (accept ? "ACCEPT" : "REJECT"),
                           resOld[0] / resOld[1],
                           resNew[0] / resNew[1],
                           (int) resOld[1], (int) resNew[1],
                           inc.norm());
                    std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() << " (rel "
                              << relAff.transpose() << ")\n";
                }
                if (accept) {
                    calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
                    resOld = resNew;
                    // JZ modification: affLight
                    aff_g2l_current = aff_g2l_new;
                    refToNew_current = refToNew_new;
                    lambda *= 0.5;
                } else {
                    lambda *= 4;
                    if (lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
                }

                if (!(inc.norm() > 1e-3)) {
                    if (debugPrint)
                        printf("inc too small, break!\n");
                    break;
                }
            }

            // set last residual for that level, as well as flow indicators.
            lastResiduals[lvl] = sqrtf((float) (resOld[0] / resOld[1]));
            lastFlowIndicators = resOld.segment<3>(2);
            if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl]) return false;


            if (levelCutoffRepeat > 1 && !haveRepeated) {
                lvl++;
                haveRepeated = true;
                printf("REPEAT LEVEL!\n");
            }
        }

        // set!
        lastToNew_out = refToNew_current;
        // JZ modification: affLight
        aff_g2l_out = aff_g2l_current;

        // JZ modification: affLight (res)
        if (!IR_Data) {
            if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
                || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
                return false;
        }
        else {
            if (ir_correction_model == 0) {
                // ideal model
                if ((setting_affineOptModeA != 0 && (fabsf((float) aff_g2l_out.a) > 2))
                    || (setting_affineOptModeB != 0 && (fabsf((float) aff_g2l_out.b) > 2)))
                    return false;
            }
            else if (ir_correction_model == 1) {
                // non-linear exponential model
                if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 200))
                    || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 1.5)))
                    return false;
            }
            else {
                // linear model
                if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 200))
                    || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
                    return false;
            }

        }

        // JZ modification: affLight
        //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();
        Vec4f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRefPrev->ab_exposure, prevFrame->ab_exposure,
                                                   lastRef_aff_g2l, aff_g2l_out, IR_Data).cast<float>();

        // JZ modification: affLight (res)
        if (!IR_Data) {
            if ((setting_affineOptModeA == 0 && (fabsf(logf((float) relAff[0])) > 1.5))
                || (setting_affineOptModeB == 0 && (fabsf((float) relAff[1]) > 200)))
                return false;
        }
        else {
            if (ir_correction_model == 0) {
                // ideal model
                if ((setting_affineOptModeA == 0 && (fabsf((float) relAff[0]) > 2))
                    || (setting_affineOptModeB == 0 && (fabsf((float) relAff[1]) > 2)))
                    return false;
            }
            else if (ir_correction_model == 1) {
                // non-linear exponential model
                if ((setting_affineOptModeA == 0 && (fabsf((float) relAff[0]) > 200))
                    || (setting_affineOptModeB == 0 && (fabsf((float) relAff[1]) > 1.5)))
                    return false;
            }
            else {
                // linear model
                if ((setting_affineOptModeA == 0 && (fabsf((float) relAff[0]) > 200))
                    || (setting_affineOptModeB == 0 && (fabsf((float) relAff[1]) > 200)))
                    return false;
            }
        }


        // JZ modification: affLight
        if(setting_affineOptModeA < 0) aff_g2l_out.a=0;
        if(setting_affineOptModeB < 0) aff_g2l_out.b=0;

        return true;
    }


    // JZ modifications: added framehessian as an argument to access the frame ID of the current keyframe
    void
    CoarseTracker::debugPlotIDepthMap(FrameHessian *fh, float *minID_pt, float *maxID_pt, std::vector<IOWrap::Output3DWrapper *> &wraps) {
        if (w[1] == 0) return;


        int lvl = 0;

        {
            std::vector<float> allID;
            for (int i = 0; i < h[lvl] * w[lvl]; i++) {
                if (idepth[lvl][i] > 0)
                    allID.push_back(idepth[lvl][i]);
            }
            std::sort(allID.begin(), allID.end());
            int n = allID.size() - 1;

            float minID_new = allID[(int) (n * 0.05)];
            float maxID_new = allID[(int) (n * 0.95)];

            float minID, maxID;
            minID = minID_new;
            maxID = maxID_new;
            if (minID_pt != 0 && maxID_pt != 0) {
                if (*minID_pt < 0 || *maxID_pt < 0) {
                    *maxID_pt = maxID;
                    *minID_pt = minID;
                } else {

                    // slowly adapt: change by maximum 10% of old span.
                    float maxChange = 0.3 * (*maxID_pt - *minID_pt);

                    if (minID < *minID_pt - maxChange)
                        minID = *minID_pt - maxChange;
                    if (minID > *minID_pt + maxChange)
                        minID = *minID_pt + maxChange;


                    if (maxID < *maxID_pt - maxChange)
                        maxID = *maxID_pt - maxChange;
                    if (maxID > *maxID_pt + maxChange)
                        maxID = *maxID_pt + maxChange;

                    *maxID_pt = maxID;
                    *minID_pt = minID;
                }
            }


            MinimalImageB3 mf(w[lvl], h[lvl]);
            mf.setBlack();


            for (int i = 0; i < h[lvl] * w[lvl]; i++) {
                int c = lastRef->dIp[lvl][i][0] * 0.9f;
                if (c > 255) c = 255;
                mf.at(i) = Vec3b(c, c, c);
                // JZ modification: dump measurements
                if (dumpMeasures) {
                    if (i == (int) (0.5 * h[lvl] * w[lvl])) {
                        char buf2[1000];
                        snprintf(buf2, 1000, "%05d", lastRef->shell->id);
                        std::ofstream out;
                        out.open("center_pixel.txt", std::ios::app | std::ios::out);
                        if (out.is_open())
                            out << buf2 << " " << c << std::endl;
                        out.close();
                    }
                }
            }

            int wl = w[lvl];
            int edgePoint = 0;
            int totalPoint = 0;
            for (int y = 3; y < h[lvl] - 3; y++)
                for (int x = 3; x < wl - 3; x++) {
                    int idx = x + y * wl;
                    float sid = 0, nid = 0;
                    float *bp = idepth[lvl] + idx;

                    if (bp[0] > 0) {
                        sid += bp[0];
                        nid++;
                    }
                    if (bp[1] > 0) {
                        sid += bp[1];
                        nid++;
                    }
                    if (bp[-1] > 0) {
                        sid += bp[-1];
                        nid++;
                    }
                    if (bp[wl] > 0) {
                        sid += bp[wl];
                        nid++;
                    }
                    if (bp[-wl] > 0) {
                        sid += bp[-wl];
                        nid++;
                    }

                    if (bp[0] > 0 || nid >= 3) {
                        float id = ((sid / nid) - minID) / ((maxID - minID));
                        mf.setPixelCirc(x, y, makeJet3B(id));
                        if (x < wl * 0.2 || x > wl * 0.8 || y < h[lvl] * 0.2 || y > h[lvl] * 0.8) {
                            edgePoint++;
                        }
                        //mf.at(idx) = makeJet3B(id);
                        totalPoint++;
                    }
                }
            //IOWrap::displayImage("coarseDepth LVL0", &mf, false);


            for (IOWrap::Output3DWrapper *ow: wraps)
                ow->pushDepthImage(&mf);

            reportKeyframeTime(fh);

            if (debugSaveImages) {
                char buf[1000];
                snprintf(buf, 1000, "images_out/predicted_%05d.png", lastRef->shell->id);
                IOWrap::writeImage(buf, &mf);
            }

            // JZ modification: dump measurements
            if (dumpMeasures) {
//                MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
//                float avgDepth = 0;
//                float validPixel = 0;
//                for (int i = 0; i < h[lvl] * w[lvl]; i++) {
//                    if ((0 <mim.at(i)) and(mim.at(i) < 100)) {
//                        avgDepth += (1.0f/mim.at(i));
//                        validPixel++;
//                    }
//                }

                char buf2[1000];
                snprintf(buf2, 1000, "%05d", lastRef->shell->id);

                float edgePercent = (float) edgePoint / (float) totalPoint;
                std::ofstream file2;
                file2.open("edge_percent.txt", std::ios::app | std::ios::out);
                if (file2.is_open())
                    file2 << buf2 << " " << edgePoint << " " << totalPoint << " " << edgePercent << std::endl;
                file2.close();
            }
        }
    }

    void CoarseTracker::reportKeyframeTime(FrameHessian *fh) {
        boost::unique_lock<boost::mutex> lk(mappingTimingMutex);

        struct timeval time_now;
        gettimeofday(&time_now, NULL);
        lastNMappingMs.push_back(
                ((time_now.tv_sec - last_map.tv_sec) * 1000.0f + (time_now.tv_usec - last_map.tv_usec) / 1000.0f));
        if (lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
        last_map = time_now;

        std::string filename("DSO_keyframe_time_ms.txt");
        std::fstream file;
        file.open(filename, std::ios_base::app | std::ios_base::in);
        if (file.is_open())
            file << fh->shell->incoming_id << " | " << lastNMappingMs.back() << std::endl;
    }

    void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper *> &wraps) {
        if (w[1] == 0) return;
        int lvl = 0;
        MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
        for (IOWrap::Output3DWrapper *ow: wraps)
            ow->pushDepthImageFloat(&mim, lastRef);
    }


    CoarseDistanceMap::CoarseDistanceMap(int ww, int hh) {
        fwdWarpedIDDistFinal = new float[ww * hh / 4];

        bfsList1 = new Eigen::Vector2i[ww * hh / 4];
        bfsList2 = new Eigen::Vector2i[ww * hh / 4];

        int fac = 1 << (pyrLevelsUsed - 1);


        coarseProjectionGrid = new PointFrameResidual *[2048 * (ww * hh / (fac * fac))];
        coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

        w[0] = h[0] = 0;
    }

    CoarseDistanceMap::~CoarseDistanceMap() {
        delete[] fwdWarpedIDDistFinal;
        delete[] bfsList1;
        delete[] bfsList2;
        delete[] coarseProjectionGrid;
        delete[] coarseProjectionGridNum;
    }


    void CoarseDistanceMap::makeDistanceMap(
            std::vector<FrameHessian *> frameHessians,
            FrameHessian *frame) {
        int w1 = w[1];
        int h1 = h[1];
        int wh1 = w1 * h1;
        for (int i = 0; i < wh1; i++)
            fwdWarpedIDDistFinal[i] = 1000;


        // make coarse tracking templates for latstRef.
        int numItems = 0;

        for (FrameHessian *fh: frameHessians) {
            if (frame == fh) continue;

            SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
            Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
            Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

            for (PointHessian *ph: fh->pointHessians) {
                assert(ph->status == PointHessian::ACTIVE);
                Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
                int u = ptp[0] / ptp[2] + 0.5f;
                int v = ptp[1] / ptp[2] + 0.5f;
                if (!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
                fwdWarpedIDDistFinal[u + w1 * v] = 0;
                bfsList1[numItems] = Eigen::Vector2i(u, v);
                numItems++;
            }
//            std::string filename("active_points.txt");
//            std::fstream file;
//            file.open(filename, std::ios_base::app | std::ios_base::in);
//        
//            if (file.is_open())
//                file << fh->pointHessians.size() << std::endl;
//            std::cout << fh->pointHessians.size() << " active points in the current keyframe." << std::endl;
        }

        growDistBFS(numItems);
    }


    void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian *> frameHessians) {

    }


    void CoarseDistanceMap::growDistBFS(int bfsNum) {
        assert(w[0] != 0);
        int w1 = w[1], h1 = h[1];
        for (int k = 1; k < 40; k++) {
            int bfsNum2 = bfsNum;
            std::swap<Eigen::Vector2i *>(bfsList1, bfsList2);
            bfsNum = 0;

            if (k % 2 == 0) {
                for (int i = 0; i < bfsNum2; i++) {
                    int x = bfsList2[i][0];
                    int y = bfsList2[i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (fwdWarpedIDDistFinal[idx + 1] > k) {
                        fwdWarpedIDDistFinal[idx + 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1] > k) {
                        fwdWarpedIDDistFinal[idx - 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + w1] > k) {
                        fwdWarpedIDDistFinal[idx + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - w1] > k) {
                        fwdWarpedIDDistFinal[idx - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }
                }
            } else {
                for (int i = 0; i < bfsNum2; i++) {
                    int x = bfsList2[i][0];
                    int y = bfsList2[i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (fwdWarpedIDDistFinal[idx + 1] > k) {
                        fwdWarpedIDDistFinal[idx + 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1] > k) {
                        fwdWarpedIDDistFinal[idx - 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + w1] > k) {
                        fwdWarpedIDDistFinal[idx + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - w1] > k) {
                        fwdWarpedIDDistFinal[idx - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }

                    if (fwdWarpedIDDistFinal[idx + 1 + w1] > k) {
                        fwdWarpedIDDistFinal[idx + 1 + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1 + w1] > k) {
                        fwdWarpedIDDistFinal[idx - 1 + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1 - w1] > k) {
                        fwdWarpedIDDistFinal[idx - 1 - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y - 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + 1 - w1] > k) {
                        fwdWarpedIDDistFinal[idx + 1 - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y - 1);
                        bfsNum++;
                    }
                }
            }
        }
    }


    void CoarseDistanceMap::addIntoDistFinal(int u, int v) {
        if (w[0] == 0) return;
        bfsList1[0] = Eigen::Vector2i(u, v);
        fwdWarpedIDDistFinal[u + w[1] * v] = 0;
        growDistBFS(1);
    }


    void CoarseDistanceMap::makeK(CalibHessian *HCalib) {
        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }

}
