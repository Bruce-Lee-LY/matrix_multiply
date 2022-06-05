// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:51:47 on Sat, Jun 04, 2022
//
// Description: matrix multiply cublas on nvidia gpu

#ifndef __MATRIX_MULTIPLY_CUDA_CUBLAS_H__
#define __MATRIX_MULTIPLY_CUDA_CUBLAS_H__

#include "cuda_trace_profile.h"

// A (M * K) * B (K * N) = C (M * N)
template <typename T>
cudaError_t matrix_multiply_cuda_cublas(const T *A, const T *B, T *C, size_t M, size_t N, size_t K) {
    if (!A || !B || !C) {
        return cudaErrorInvalidDevicePointer;
    }

    if (M == 0 || N == 0 || K == 0) {
        return cudaErrorInvalidValue;
    }

    cublasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MATRIX_CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    {
        MATRIX_CUDA_TRACE_PROFILE(__FUNCTION__);
        MATRIX_CHECK_CUBLAS_ERROR(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M));
    }

    MATRIX_CHECK_CUBLAS_ERROR(cublasDestroy(handle));

    return cudaGetLastError();
}

#endif  // __MATRIX_MULTIPLY_CUDA_CUBLAS_H__
