cmake_minimum_required(VERSION 3.14)
project(IJRR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==== Compile flags ====
add_compile_options(
  -O3 -march=native -funroll-loops -ffast-math
  -DCASADI_REAL=double -Dcasadi_real=double
)

# ==== Dependencies ====
find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(LieIPM REQUIRED)

find_path(MATIO_INCLUDE_DIR matio.h)
find_library(MATIO_LIBRARY matio)
if(NOT MATIO_INCLUDE_DIR OR NOT MATIO_LIBRARY)
    message(FATAL_ERROR "matio not found")
endif()

# ==== Header-only LieIPM Interface Library ====
add_library(LieIPMIface INTERFACE)
target_include_directories(LieIPMIface
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../LieIPM-cpp/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gen>
    ${MATIO_INCLUDE_DIR}
)
target_link_libraries(LieIPMIface
  INTERFACE
    Eigen3::Eigen
    ${MATIO_LIBRARY}
    dmumps_seq mumps_common_seq
    smumps_seq
    openblas
)

# ==== CasADi generated sources ====
file(GLOB GEN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/gen/*.cpp")

# ==== exp_traj executable ====
add_executable(exp_traj
  src/exp_traj.cpp
  ${GEN_SOURCES}
)
target_link_libraries(exp_traj PRIVATE LieIPMIface)