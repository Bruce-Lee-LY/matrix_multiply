# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:56:32 on Thu, Jun 02, 2022
#
# Description: cmake for matrix multiply

cmake_minimum_required (VERSION 3.8)

project (matrix_multiply LANGUAGES CXX)

set (CMAKE_VERBOSE_MAKEFILE ${MATRIX_VERBOSE_MAKEFILE})

set (CMAKE_CXX_FLAGS "-std=c++11")
set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -DMATRIX_BUILD_DEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

add_compile_options (
    -Wall
    -Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

include_directories (
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/cpu
    ${PROJECT_SOURCE_DIR}/src/cuda
)

# cpu
add_executable (matrix_multiply_cpu ${PROJECT_SOURCE_DIR}/src/cpu/cpu.cpp)

install (TARGETS matrix_multiply_cpu RUNTIME DESTINATION cpu)

# nvidia gpu
find_package (CUDA REQUIRED)
unset (CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
option (CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set (CUDA_VERBOSE_BUILD ${MATRIX_VERBOSE_MAKEFILE})
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set (SYS_CUDART_PATH "/usr/local/cuda/targets/x86_64-linux")
set (SYS_CUDA_DRIVER_PATH "/usr/lib/x86_64-linux-gnu")

include_directories (
    ${SYS_CUDART_PATH}/include
)

link_directories (
    ${SYS_CUDART_PATH}/lib
    ${SYS_CUDA_DRIVER_PATH}
)

cuda_add_executable (matrix_multiply_cuda ${PROJECT_SOURCE_DIR}/src/cuda/cuda.cu)
target_link_libraries (matrix_multiply_cuda -lcublas)

install (TARGETS matrix_multiply_cuda RUNTIME DESTINATION cuda)
