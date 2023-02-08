// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:51:47 on Sat, Jun 04, 2022
//
// Description: matrix multiply shared memory on nvidia gpu

#ifndef __MATRIX_MULTIPLY_CUDA_SHARED_MEMORY_H__
#define __MATRIX_MULTIPLY_CUDA_SHARED_MEMORY_H__

#include "cuda_trace_profile.h"

#define MATRIX_CUDA_SM_BLOCK_SIZE 32

template <typename T>
__global__ void matrix_multiply_shared_memory(const T *A, const T *B, T *C, size_t M, size_t N, size_t K) {
    size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ T share_A[MATRIX_CUDA_SM_BLOCK_SIZE][MATRIX_CUDA_SM_BLOCK_SIZE];
    __shared__ T share_B[MATRIX_CUDA_SM_BLOCK_SIZE][MATRIX_CUDA_SM_BLOCK_SIZE];

    T t = 0.0, y = 0.0, r = 0.0;
    for (size_t i = 0; i < (K - 1) / MATRIX_CUDA_SM_BLOCK_SIZE + 1; ++i) {
        share_A[threadIdx.y][threadIdx.x] = A[row * K + (i * MATRIX_CUDA_SM_BLOCK_SIZE + threadIdx.x)];
        share_B[threadIdx.y][threadIdx.x] = B[(i * MATRIX_CUDA_SM_BLOCK_SIZE + threadIdx.y) * N + column];
        __syncthreads();

        for (size_t j = 0; j < MATRIX_CUDA_SM_BLOCK_SIZE; ++j) {
            y -= share_A[threadIdx.y][j] * share_B[j][threadIdx.x];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
        __syncthreads();
    }

    if (column < N && row < M) {
        C[row * N + column] = t;
    }
}

// A (M * K) * B (K * N) = C (M * N)
template <typename T>
cudaError_t matrix_multiply_cuda_shared_memory(const T *A, const T *B, T *C, size_t M, size_t N, size_t K,
                                               size_t block) {
    if (!A || !B || !C) {
        return cudaErrorInvalidDevicePointer;
    }

    if (M == 0 || N == 0 || K == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 block_dim(block, block);
    const dim3 grid_dim(div_ceil(N, block_dim.x), div_ceil(M, block_dim.y));

    {
        MATRIX_CUDA_TRACE_PROFILE("matrix_multiply_cuda_shared_memory" + std::string("_") + std::to_string(block));
        matrix_multiply_shared_memory<T><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    }

    return cudaGetLastError();
}

#endif  // __MATRIX_MULTIPLY_CUDA_SHARED_MEMORY_H__
