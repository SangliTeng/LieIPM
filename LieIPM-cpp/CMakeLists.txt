cmake_minimum_required(VERSION 3.14)
project(lieipm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==== Compile flags ====
add_compile_options(
  -O3 -march=native -funroll-loops -ffast-math
  -DCASADI_REAL=double -Dcasadi_real=double
)

# ==== Dependencies ====
find_package(Eigen3 REQUIRED)
set(OpenBLAS_DIR "/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake/openblas" CACHE PATH "OpenBLAS config path")
find_package(OpenBLAS REQUIRED)

find_library(MUMPS_DMUMPS_LIBRARY       dmumps       HINTS /usr/local/lib)
find_library(MUMPS_MUMPS_COMMON_LIBRARY mumps_common HINTS /usr/local/lib)
find_library(MUMPS_PORD_LIBRARY         pord         HINTS /usr/local/lib)
find_library(MUMPS_MPISEQ_LIBRARY       mpiseq       HINTS /usr/local/lib)

# ==== Header-only LieIPM Interface Library ====
add_library(LieIPM INTERFACE)
target_include_directories(LieIPM
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(LieIPM
  INTERFACE
    Eigen3::Eigen
    ${MUMPS_DMUMPS_LIBRARY}
    ${MUMPS_MUMPS_COMMON_LIBRARY}
    ${MUMPS_PORD_LIBRARY}
    ${MUMPS_MPISEQ_LIBRARY}
    openblas
    gfortran m
)

# ==== Install support ====
install(DIRECTORY include/ DESTINATION include)
install(TARGETS LieIPM EXPORT LieIPMTargets)
install(EXPORT LieIPMTargets
        FILE LieIPMConfig.cmake
        NAMESPACE LieIPM::
        DESTINATION lib/cmake/LieIPM)