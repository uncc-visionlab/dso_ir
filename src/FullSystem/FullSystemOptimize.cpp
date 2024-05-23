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



#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ResidualProjections.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso {


    void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual *> *toRemove, int min,
                                           int max, Vec10 *stats, int tid) {
        int cnt = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;
        for (int k = min; k < max; k++) {
            PointFrameResidual *r = activeResiduals[k];
//            std::cout << "r->host.a = " << r->host->aff_g2l().a << "; r->host.b = " << r->host->aff_g2l().b << std::endl;
            (*stats)[0] += r->linearize(&Hcalib);       // set the state of the residual (IN or OOB)
            if((r->state_state == ResState::OOB)) {
                cnt1++;
            }
            else{
                if((r->state_NewState == ResState::IN))
                    cnt2++;
                else if ((r->state_NewState == ResState::OUTLIER))
                    cnt3++;
                else if ((r->state_NewState == ResState::OOB))
                    cnt4++;
            }
            if (fixLinearization) {
                r->applyRes(true);      // set the status of efResidual (active or not)

                if (r->efResidual->isActive()) {
                    cnt++;
                    if (r->isNew) {
                        PointHessian *p = r->point;
                        Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll *
                                        Vec3f(p->u, p->v, 1);    // projected point assuming infinite depth.
                        Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll *
                                              p->idepth_scaled;    // projected point with real depth.
                        float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) -
                                              (ptp.head<2>() / ptp[2])).norm();    // 0.01 = one pixel.


                        if (relBS > p->maxRelBaseline)
                            p->maxRelBaseline = relBS;

                        p->numGoodResiduals++;
                    }
                } else {
                    toRemove[tid].push_back(activeResiduals[k]);
                }
            }
        }
//            std::cout << "active residuals = " << max << "; linearized residual = " << cnt << "; old OOB residual = " << cnt1 <<
//            "; IN residual = " << cnt2 << "; OUTLIER residual = " << cnt3 << "; new OOB residual = " << cnt4 << std::endl;
    }


    void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid) {
        for (int k = min; k < max; k++)
            activeResiduals[k]->applyRes(true);
    }

    void FullSystem::setNewFrameEnergyTH() {

        // collect all residuals and make decision on TH.
        allResVec.clear();
        allResVec.reserve(activeResiduals.size() * 2);
        FrameHessian *newFrame = frameHessians.back();

        for (PointFrameResidual *r: activeResiduals)
            if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) {
                allResVec.push_back(r->state_NewEnergyWithOutlier);

            }

        if (allResVec.size() == 0) {
            newFrame->frameEnergyTH = 12 * 12 * patternNum;
            return;        // should never happen, but lets make sure.
        }


        int nthIdx = setting_frameEnergyTHN * allResVec.size();

        assert(nthIdx < (int) allResVec.size());
        assert(setting_frameEnergyTHN < 1);

        std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
        float nthElement = sqrtf(allResVec[nthIdx]);


        newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
        newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight +
                                  newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
        newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
        newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;



//
//	int good=0,bad=0;
//	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
//	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
//			meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
//			good, bad);
    }

    Vec3 FullSystem::linearizeAll(bool fixLinearization) {
        double lastEnergyP = 0;     //  point energy (Huber weighted residual)
        double lastEnergyR = 0;
        double num = 0;


        std::vector<PointFrameResidual *> toRemove[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++) toRemove[i].clear();

        if (multiThreading) {
            treadReduce.reduce(
                    boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4),
                    0, activeResiduals.size(), 0);
            lastEnergyP = treadReduce.stats[0];
        } else {
            Vec10 stats;
            linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
            lastEnergyP = stats[0];
        }


        setNewFrameEnergyTH();


        if (fixLinearization) {

            for (PointFrameResidual *r: activeResiduals) {
                PointHessian *ph = r->point;
                if (ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].second = r->state_state;
                else if (ph->lastResiduals[1].first == r)
                    ph->lastResiduals[1].second = r->state_state;


            }

            int nResRemoved = 0;
            for (int i = 0; i < NUM_THREADS; i++) {
                for (PointFrameResidual *r: toRemove[i]) {
                    PointHessian *ph = r->point;

                    if (ph->lastResiduals[0].first == r)
                        ph->lastResiduals[0].first = 0;
                    else if (ph->lastResiduals[1].first == r)
                        ph->lastResiduals[1].first = 0;

                    for (unsigned int k = 0; k < ph->residuals.size(); k++)
                        if (ph->residuals[k] == r) {
                            ef->dropResidual(r->efResidual);
                            deleteOut<PointFrameResidual>(ph->residuals, k);
                            nResRemoved++;
                            break;
                        }
                }
            }
            //printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved, (int)activeResiduals.size());

        }

        return Vec3(lastEnergyP, lastEnergyR, num);
    }


// applies step to linearization point.
    bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD) {
//	float meanStepC=0,meanStepP=0,meanStepD=0;
//	meanStepC += Hcalib.step.norm();

        Vec10 pstepfac;
        pstepfac.segment<3>(0).setConstant(stepfacT);
        pstepfac.segment<3>(3).setConstant(stepfacR);
        pstepfac.segment<4>(6).setConstant(stepfacA);


        float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

        float sumNID = 0;
        if (setting_solverMode & SOLVER_MOMENTUM) {     // not being executed
            Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
            for (FrameHessian *fh: frameHessians) {
                Vec10 step = fh->step;
                step.head<6>() += 0.5f * (fh->step_backup.head<6>());

                fh->setState(fh->state_backup + step);
                sumA += step[6] * step[6];
                sumB += step[7] * step[7];
                sumT += step.segment<3>(0).squaredNorm();
                sumR += step.segment<3>(3).squaredNorm();

                for (PointHessian *ph: fh->pointHessians) {
                    float step = ph->step + 0.5f * (ph->step_backup);
                    ph->setIdepth(ph->idepth_backup + step);
                    sumID += step * step;
                    sumNID += fabsf(ph->idepth_backup);
                    numID++;

                    ph->setIdepthZero(ph->idepth_backup + step);
                }
            }
        } else {    // executed
            Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);
            for (FrameHessian *fh: frameHessians) {
//                std::cout << "before = " << fh->state[0] << ", " << fh->state[1] << ", " << fh->state[2] << ", " << fh->state[3] << ", " << fh->state[4] << ", "
//                          << fh->state[5] << ", " << fh->state[6] << ", " << fh->state[7] << std::endl;
//                Vec10 step = pstepfac.cwiseProduct(fh->step);
//                std::cout << "pstepfac = " << pstepfac[0] << ", " << pstepfac[1] << ", " << pstepfac[2] << ", " << pstepfac[3] << ", " << pstepfac[4] << ", "
//                          << pstepfac[5] << ", " << pstepfac[6] << ", " << pstepfac[7] << std::endl;
//                std::cout << "fh->step = " << fh->step[0] << ", " << fh->step[1] << ", " << fh->step[2] << ", " << fh->step[3] << ", " << fh->step[4] << ", "
//                          << fh->step[5] << ", " << fh->step[6] << ", " << fh->step[7] << std::endl;
//                std::cout << "step = " << step[0] << ", " << step[1] << ", " << step[2] << ", " << step[3] << ", " << step[4] << ", " << step[5] << ", " << step[6] << ", " << step[7] << std::endl;
                fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
//                std::cout << "after = " << fh->state[0] << ", " << fh->state[1] << ", " << fh->state[2] << ", " << fh->state[3] << ", " << fh->state[4] << ", "
//                          << fh->state[5] << ", " << fh->state[6] << ", " << fh->state[7] << std::endl;
                sumA += fh->step[6] * fh->step[6];
                sumB += fh->step[7] * fh->step[7];
                sumT += fh->step.segment<3>(0).squaredNorm();
                sumR += fh->step.segment<3>(3).squaredNorm();

                for (PointHessian *ph: fh->pointHessians) {
                    ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
                    sumID += ph->step * ph->step;
                    sumNID += fabsf(ph->idepth_backup);
                    numID++;

                    ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
                }
            }
        }
        sumA /= frameHessians.size();
        sumB /= frameHessians.size();
        sumR /= frameHessians.size();
        sumT /= frameHessians.size();
        sumID /= numID;
        sumNID /= numID;


        if (!setting_debugout_runquiet)
            printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
                   sqrtf(sumA) / (0.0005 * setting_thOptIterations),
                   sqrtf(sumB) / (0.00005 * setting_thOptIterations),
                   sqrtf(sumR) / (0.00005 * setting_thOptIterations),
                   sqrtf(sumT) * sumNID / (0.00005 * setting_thOptIterations));


        EFDeltaValid = false;
        setPrecalcValues();

        return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
               sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
//
//	printf("mean steps: %f %f %f!\n",
//			meanStepC, meanStepP, meanStepD);
    }


// sets linearization point.
    void FullSystem::backupState(bool backupLastStep) {
        if (setting_solverMode & SOLVER_MOMENTUM) {
            if (backupLastStep) {
                Hcalib.step_backup = Hcalib.step;
                Hcalib.value_backup = Hcalib.value;
                for (FrameHessian *fh: frameHessians) {
                    fh->step_backup = fh->step;
                    fh->state_backup = fh->get_state();
                    for (PointHessian *ph: fh->pointHessians) {
                        ph->idepth_backup = ph->idepth;
                        ph->step_backup = ph->step;
                    }
                }
            } else {
                Hcalib.step_backup.setZero();
                Hcalib.value_backup = Hcalib.value;
                for (FrameHessian *fh: frameHessians) {
                    fh->step_backup.setZero();
                    fh->state_backup = fh->get_state();
                    for (PointHessian *ph: fh->pointHessians) {
                        ph->idepth_backup = ph->idepth;
                        ph->step_backup = 0;
                    }
                }
            }
        } else {
            Hcalib.value_backup = Hcalib.value;
            for (FrameHessian *fh: frameHessians) {
                fh->state_backup = fh->get_state();
                for (PointHessian *ph: fh->pointHessians)
                    ph->idepth_backup = ph->idepth;
            }
        }
    }

// sets linearization point.
    void FullSystem::loadSateBackup() {
        Hcalib.setValue(Hcalib.value_backup);
        for (FrameHessian *fh: frameHessians) {
            fh->setState(fh->state_backup);
            for (PointHessian *ph: fh->pointHessians) {
                ph->setIdepth(ph->idepth_backup);

                ph->setIdepthZero(ph->idepth_backup);
            }

        }


        EFDeltaValid = false;
        setPrecalcValues();
    }


    double FullSystem::calcMEnergy() {
        if (setting_forceAceptStep) return 0;
        // calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
        //ef->makeIDX();
        //ef->setDeltaF(&Hcalib);
        return ef->calcMEnergyF();

    }


    void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a,
                                 float b) {
        //printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
        printf("Energy_Active = %f, Avg_Energy_Active=%.3f. Point_Num: A(%'d) + M(%'d); ab = (%f %f)!\n",
               res[0],
               sqrtf((float) (res[0] / (patternNum * ef->resInA))),
               ef->resInA,
               ef->resInM,
               a,
               b
        );

    }


    float FullSystem::optimize(int mnumOptIts) {

        if (frameHessians.size() < 2) return 0;
        if (frameHessians.size() < 3) mnumOptIts = 20;
        if (frameHessians.size() < 4) mnumOptIts = 15;






        // get statistics and active residuals.

        activeResiduals.clear();
        int numPoints = 0;
        int numLRes = 0;
        for (FrameHessian *fh: frameHessians)
            for (PointHessian *ph: fh->pointHessians) {
                for (PointFrameResidual *r: ph->residuals) {
                    if (!r->efResidual->isLinearized) {
                        activeResiduals.push_back(r);
                        r->resetOOB();
                    } else
                        numLRes++;
                }
                numPoints++;
            }

        if (!setting_debugout_runquiet)
            printf("OPTIMIZE %d pts, %d active res, %d lin res!\n", ef->nPoints, (int) activeResiduals.size(), numLRes);


        Vec3 lastEnergy = linearizeAll(false);      // lastEnergy[0] is the point energy (Huber weighted residual). The rest of the vector is never be assigned nor used.
        double lastEnergyL = calcLEnergy();
        double lastEnergyM = calcMEnergy();


        if (multiThreading)
            treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                               activeResiduals.size(), 50);
        else
            applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);


        if (!setting_debugout_runquiet) {
            printf("Initial Error (Energy): ");
            // JZ modification: affLight
            printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
        }

        debugPlotTracking();

        double lambda = 1e-1;
        float stepsize = 1;
        VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);
        for (int iteration = 0; iteration < mnumOptIts; iteration++) {
            // solve!
            backupState(iteration != 0);
            //solveSystemNew(0);
            solveSystem(iteration, lambda);     // solve for the incremental step of each parameter in a frameHessian
            double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
            previousX = ef->lastX;


            if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
                float newStepsize = exp(incDirChange * 1.4);
                if (incDirChange < 0 && stepsize > 1) stepsize = 1;

                stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
                if (stepsize > 2) stepsize = 2;
                if (stepsize < 0.25) stepsize = 0.25;
            }
            bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);




            // eval new energy!
            Vec3 newEnergy = linearizeAll(false);
            double newEnergyL = calcLEnergy();
            double newEnergyM = calcMEnergy();


            if (!setting_debugout_runquiet) {
                printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
                       (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                        lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
                       iteration,
                       log10(lambda),
                       incDirChange,
                       stepsize);
                // JZ modification: affLight
                printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
                // JZ modification: dump measurements
                if (dumpMeasures) {
                    if ((newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                         lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) && (iteration == (mnumOptIts - 1))) {
                        std::ofstream out;
                        out.open("ab_values.txt", std::ios::app | std::ios::out);
                        //std::fstream file;
                        //file.open(filename, std::ios_base::app | std::ios_base::in);
                        if (out.is_open())
                            out << frameHessians.back()->aff_g2l().a << " " << frameHessians.back()->aff_g2l().b << std::endl;
                        out.close();
                    }
                }
            }

            if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                                           lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {

                if (multiThreading)
                    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                                       activeResiduals.size(), 50);
                else
                    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

                lastEnergy = newEnergy;
                lastEnergyL = newEnergyL;
                lastEnergyM = newEnergyM;

                lambda *= 0.25;
            } else {
                loadSateBackup();
                lastEnergy = linearizeAll(false);
                lastEnergyL = calcLEnergy();
                lastEnergyM = calcMEnergy();
                lambda *= 1e2;
            }


		if(canbreak && iteration >= setting_minOptIterations) break;
	}



        Vec10 newStateZero = Vec10::Zero();
        newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

        frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
                                        newStateZero);
        EFDeltaValid = false;
        EFAdjointsValid = false;
        ef->setAdjointsF(&Hcalib);
        setPrecalcValues();

        lastEnergy = linearizeAll(true);


        if (!std::isfinite((double) lastEnergy[0]) || !std::isfinite((double) lastEnergy[1]) ||
            !std::isfinite((double) lastEnergy[2])) {
            printf("KF Tracking failed: LOST!\n");
            isLost = true;
        }


        statistics_lastFineTrackRMSE = sqrtf((float) (lastEnergy[0] / (patternNum * ef->resInA)));

        if (calibLog != 0) {
            (*calibLog) << Hcalib.value_scaled.transpose() <<
                        " " << frameHessians.back()->get_state_scaled().transpose() <<
                        " " << sqrtf((float) (lastEnergy[0] / (patternNum * ef->resInA))) <<
                        " " << ef->resInM << "\n";
            calibLog->flush();
        }

        {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            for (FrameHessian *fh: frameHessians) {
                fh->shell->camToWorld = fh->PRE_camToWorld;
                // JZ modification: affLight
                fh->shell->aff_g2l = fh->aff_g2l();
            }
        }


        debugPlotTracking();

        printf("Optimized camera parameters are fx = %f, fy = %f, cx = %f, cy = %f.\n", Hcalib.fxl(), Hcalib.fyl(),
               Hcalib.cxl(), Hcalib.cyl());

        std::cout << "lastEnergy = " << lastEnergy[0] << "; patternNum * ef->resInA = " << patternNum * ef->resInA << std::endl;

        return sqrtf((float) (lastEnergy[0] / (patternNum * ef->resInA)));

    }


    void FullSystem::solveSystem(int iteration, double lambda) {
        ef->lastNullspaces_forLogging = getNullspaces(
                ef->lastNullspaces_pose,
                ef->lastNullspaces_scale,
                ef->lastNullspaces_affA,
                ef->lastNullspaces_affB);   // JZ modification: affLight
        ef->solveSystemF(iteration, lambda, &Hcalib);       // solve for the incremental step of each parameter in a frameHessian
    }


    double FullSystem::calcLEnergy() {
        if (setting_forceAceptStep) return 0;

        double Ef = ef->calcLEnergyF_MT();
        return Ef;

    }


    void FullSystem::removeOutliers() {
        int numPointsDropped = 0;
        for (FrameHessian *fh: frameHessians) {
            for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
                PointHessian *ph = fh->pointHessians[i];
                if (ph == 0) continue;

                if (ph->residuals.size() == 0) {
                    fh->pointHessiansOut.push_back(ph);
                    ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                    fh->pointHessians[i] = fh->pointHessians.back();
                    fh->pointHessians.pop_back();
                    i--;
                    numPointsDropped++;
                }
            }
        }
        ef->dropPointsF();
    }


    std::vector<VecX> FullSystem::getNullspaces(    // JZ modification: affLight
            std::vector<VecX> &nullspaces_pose,
            std::vector<VecX> &nullspaces_scale,
            std::vector<VecX> &nullspaces_affA,
            std::vector<VecX> &nullspaces_affB) {
        nullspaces_pose.clear();
        nullspaces_scale.clear();
        nullspaces_affA.clear();
        nullspaces_affB.clear();


        int n = CPARS + frameHessians.size() * 8;
        std::vector<VecX> nullspaces_x0_pre;
        for (int i = 0; i < 6; i++) {
            VecX nullspace_x0(n);
            nullspace_x0.setZero();
            for (FrameHessian *fh: frameHessians) {
                nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
                nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
                nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
            }
            nullspaces_x0_pre.push_back(nullspace_x0);
            nullspaces_pose.push_back(nullspace_x0);
        }
        for (int i = 0; i < 2; i++) {
            VecX nullspace_x0(n);
            nullspace_x0.setZero();
            for (FrameHessian *fh: frameHessians) {
                nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) = fh->nullspaces_affine.col(i).head<2>();
                nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
                nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
            }
            nullspaces_x0_pre.push_back(nullspace_x0);
            if (i == 0) nullspaces_affA.push_back(nullspace_x0);   // JZ modification: affLight
            if (i == 1) nullspaces_affB.push_back(nullspace_x0);
        }

        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (FrameHessian *fh: frameHessians) {
            nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
            nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
            nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_scale.push_back(nullspace_x0);

        return nullspaces_x0_pre;
    }

}
