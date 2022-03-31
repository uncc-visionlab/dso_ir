
cmake_minimum_required(VERSION 3.10)

message(STATUS "Configuring Pangolin...")
set(EXPORT_Pangolin OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_PYPANGOLIN_MODULE OFF CACHE BOOL "")
mark_as_advanced(EXPORT_Pangolin)
mark_as_advanced(BUILD_EXAMPLES)
mark_as_advanced(BUILD_TESTS)
mark_as_advanced(BUILD_PYPANGOLIN_MODULE)
add_subdirectory(Pangolin)
set(Pangolin_DIR ${Pangolin_BINARY_DIR}/src PARENT_SCOPE)
message(STATUS "Pangolin_BINARY_DIR: ${Pangolin_BINARY_DIR}")
message(STATUS "... Pangolin configuration done.")

# Pangolin
#add_subdirectory(Pangolin)

