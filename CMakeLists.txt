# Build adapted from the FHERMA CUDA scaffold; use the runner GPU architecture.
#
# A CUDA build: the CUDA language is enabled and the answer is compiled by nvcc.
# It differs from the plain C++ CMakeLists in three ways — the CUDA language,
# the device-LTO flags, and linking NVIDIA cuPQC when the challenge image
# provides it. Everything else is the same, and the platform builds it with the
# same `cmake -B build && cmake --build build`.
cmake_minimum_required(VERSION 3.24)
project(fherma_solution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release unless told otherwise: an answer measured at -O0 is not a measurement
# of the answer.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compile for the GPU visible on the runner; override for cross-compilation.
# This supports the RTX PRO 6000 Blackwell runner as well as L40S. CMAKE_CUDA_ARCHITECTURES is
# left off deliberately: it emits -gencode, which nvcc refuses alongside the
# -dlto that cuPQC's device-LTO objects need, so the architecture is passed as
# -arch= instead.
set(FHERMA_CUDA_ARCH native CACHE STRING "nvcc -arch value")

set(FHERMA_SOLVER wide CACHE STRING "Arithmetic backend: wide or rns")
if(FHERMA_SOLVER STREQUAL "rns")
  add_executable(solution main.cpp rns/solve.cu)
elseif(FHERMA_SOLVER STREQUAL "wide")
  add_executable(solution main.cpp solve.cu)
else()
  message(FATAL_ERROR "Unknown FHERMA_SOLVER: ${FHERMA_SOLVER}")
endif()
find_package(Threads REQUIRED)
target_link_libraries(solution PRIVATE Threads::Threads)
target_include_directories(solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(solution PROPERTIES
  CUDA_ARCHITECTURES OFF
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(solution PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${FHERMA_CUDA_ARCH};-dlto>)
target_link_options(solution PRIVATE
  $<DEVICE_LINK:-arch=${FHERMA_CUDA_ARCH};-dlto>)
option(FHERMA_RESOURCE_REPORT "Print CUDA register and spill statistics" OFF)
if(FHERMA_RESOURCE_REPORT)
  target_compile_options(solution PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  target_link_options(solution PRIVATE $<DEVICE_LINK:-Xptxas=-v>)
endif()

# NVIDIA cuPQC, when the challenge image provides it (CUPQC_SDK_DIR is set in
# fherma/cupqc): its device BigInt library and headers. BigInt is linked because
# the wide-integer kernels are what a CUDA answer here usually reaches for; add
# cupqc-ntt, cupqc-hash or cupqc-pk the same way if you use them.
if(DEFINED ENV{CUPQC_SDK_DIR})
  target_include_directories(solution PRIVATE
    $ENV{CUPQC_SDK_DIR}/include
    $ENV{CUPQC_SDK_DIR}/include/cupqc)
  target_link_libraries(solution PRIVATE
    $ENV{CUPQC_SDK_DIR}/lib/libcupqc-bigint.a)
endif()
