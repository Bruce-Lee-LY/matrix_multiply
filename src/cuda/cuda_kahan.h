// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:51:47 on Sat, Jun 04, 2022
//
// Description: matrix multiply Kahan’s Summation Formula on nvidia gpu

#ifndef __MATRIX_MULTIPLY_CUDA_KAHAN_H__
#define __MATRIX_MULTIPLY_CUDA_KAHAN_H__

#include "cuda_trace_profile.h"

template <typename T>
__global__ void matrix_multiply_kahan(const T *A, const T *B, T *C, size_t M, size_t N, size_t K) {
    size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column < N && row < M) {
        T t = 0.0, y = 0.0, r = 0.0;
        for (size_t i = 0; i < K; ++i) {
            y -= A[row * K + i] * B[i * N + column];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
        C[row * N + column] = t;
    }
}

// A (M * K) * B (K * N) = C (M * N)
// this is done by keeping a separate running compensation (a variable to accumulate small errors)
template <typename T>
cudaError_t matrix_multiply_cuda_kahan(const T *A, const T *B, T *C, size_t M, size_t N, size_t K, size_t block) {
    if (!A || !B || !C) {
        return cudaErrorInvalidDevicePointer;
    }

    if (M == 0 || N == 0 || K == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 block_dim(block, block);
    const dim3 grid_dim(div_ceil(N, block_dim.x), div_ceil(M, block_dim.y));

    {
        MATRIX_CUDA_TRACE_PROFILE("matrix_multiply_cuda_kahan" + std::string("_") + std::to_string(block));
        matrix_multiply_kahan<T><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    }

    return cudaGetLastError();
}

#endif  // __MATRIX_MULTIPLY_CUDA_KAHAN_H__
