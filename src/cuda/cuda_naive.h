// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:51:47 on Sat, Jun 04, 2022
//
// Description: matrix multiply naive on nvidia gpu

#ifndef __MATRIX_MULTIPLY_CUDA_NAIVE_H__
#define __MATRIX_MULTIPLY_CUDA_NAIVE_H__

#include "cuda_trace_profile.h"

template <typename T>
__global__ void matrix_multiply_naive(const T *A, const T *B, T *C, size_t M, size_t N, size_t K) {
    size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column < N && row < M) {
        T tmp = 0.0;
        for (size_t i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + column];
        }
        C[row * N + column] = tmp;
    }
}

// A (M * K) * B (K * N) = C (M * N)
template <typename T>
cudaError_t matrix_multiply_cuda_naive(const T *A, const T *B, T *C, size_t M, size_t N, size_t K, size_t block) {
    if (!A || !B || !C) {
        return cudaErrorInvalidDevicePointer;
    }

    if (M == 0 || N == 0 || K == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 block_dim(block, block);
    const dim3 grid_dim(div_ceil(N, block_dim.x), div_ceil(M, block_dim.y));

    {
        MATRIX_CUDA_TRACE_PROFILE("matrix_multiply_cuda_naive" + std::string("_") + std::to_string(block));
        matrix_multiply_naive<T><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    }

    return cudaGetLastError();
}

#endif  // __MATRIX_MULTIPLY_CUDA_NAIVE_H__
