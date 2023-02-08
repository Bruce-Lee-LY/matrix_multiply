// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: cuda common

#ifndef __MATRIX_MULTIPLY_CUDA_COMMON_H__
#define __MATRIX_MULTIPLY_CUDA_COMMON_H__

#include "common.h"
#include "cublas_v2.h"
#include "cuda_runtime_api.h"

#define MATRIX_CHECK_CUDART_ERROR(_expr_)                                                         \
    do {                                                                                          \
        cudaError_t _ret_ = _expr_;                                                               \
        if (_ret_ != cudaSuccess) {                                                               \
            const char *_err_str_ = cudaGetErrorName(_ret_);                                      \
            int _rt_version_ = 0;                                                                 \
            cudaRuntimeGetVersion(&_rt_version_);                                                 \
            int _driver_version_ = 0;                                                             \
            cudaDriverGetVersion(&_driver_version_);                                              \
            MLOG("CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d", \
                 static_cast<int>(_ret_), _err_str_, _rt_version_, _driver_version_);             \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define MATRIX_CHECK_CUBLAS_ERROR(_expr_)                                                                 \
    do {                                                                                                  \
        cublasStatus_t _ret_ = _expr_;                                                                    \
        if (_ret_ != CUBLAS_STATUS_SUCCESS) {                                                             \
            size_t _rt_version_ = cublasGetCudartVersion();                                               \
            MLOG("CUBLAS API error = %04d, runtime version: %zu", static_cast<int>(_ret_), _rt_version_); \
            exit(EXIT_FAILURE);                                                                           \
        }                                                                                                 \
    } while (0)

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <typename T>
void print_cuda_matrix(const T *matrix, const size_t &row, const size_t &column,
                       const std::string &name = "cuda_matrix") {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
    std::vector<std::vector<T>> tmp(row, std::vector<T>(column));
    for (size_t i = 0; i < row; ++i) {
        MATRIX_CHECK_CUDART_ERROR(
            cudaMemcpy(tmp[i].data(), matrix + i * column, column * sizeof(T), cudaMemcpyDeviceToHost));
    }
    print_matrix<T>(tmp, name);
}

#endif  // __MATRIX_MULTIPLY_CUDA_COMMON_H__
