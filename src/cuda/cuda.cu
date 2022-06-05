// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:51:47 on Sat, Jun 04, 2022
//
// Description: test matrix multiply on nvidia gpu

#include "cuda_cublas.h"
#include "cuda_kahan.h"
#include "cuda_naive.h"
#include "cuda_shared_memory.h"

#define CUDA_MATRIX_DIMENSION 512

int main() {
    MATRIX_TRACE_PROFILE("matrix_multiply_cuda");

    size_t A_row = CUDA_MATRIX_DIMENSION, A_column = CUDA_MATRIX_DIMENSION, B_row = CUDA_MATRIX_DIMENSION,
           B_column = CUDA_MATRIX_DIMENSION;
    std::vector<std::vector<float>> A(A_row, std::vector<float>(A_column));
    std::vector<std::vector<float>> B(B_row, std::vector<float>(B_column));

    float min = 1.0, max = 100.0;
    get_random_matrix<float>(A, min, max);
    // print_matrix<float>(A, "A");

    get_random_matrix<float>(B, min, max);
    // print_matrix<float>(B, "B");

    float *dev_A = nullptr, *dev_B = nullptr, *dev_C = nullptr;
    MATRIX_CHECK_CUDART_ERROR(cudaMalloc((void **)&dev_A, A_row * A_column * sizeof(float)));
    MATRIX_CHECK_CUDART_ERROR(cudaMalloc((void **)&dev_B, B_row * B_column * sizeof(float)));
    MATRIX_CHECK_CUDART_ERROR(cudaMalloc((void **)&dev_C, A_row * B_column * sizeof(float)));

    for (size_t i = 0; i < A_row; ++i) {
        MATRIX_CHECK_CUDART_ERROR(
            cudaMemcpy(dev_A + i * A_column, A[i].data(), A_column * sizeof(float), cudaMemcpyHostToDevice));
    }

    for (size_t i = 0; i < B_row; ++i) {
        MATRIX_CHECK_CUDART_ERROR(
            cudaMemcpy(dev_B + i * B_column, B[i].data(), B_column * sizeof(float), cudaMemcpyHostToDevice));
    }

    MLOG("=================== cublas ===================");
    MATRIX_CHECK_CUDART_ERROR(matrix_multiply_cuda_cublas<float>(dev_A, dev_B, dev_C, A_row, B_column, A_column));
    // print_cuda_matrix<float>(dev_C, A_row, B_column, "cublas C");

    MLOG("=================== naive ===================");
    for (size_t i = 8; i <= 32; i *= 2) {
        MATRIX_CHECK_CUDART_ERROR(matrix_multiply_cuda_naive<float>(dev_A, dev_B, dev_C, A_row, B_column, A_column, i));
        // print_cuda_matrix<float>(dev_C, A_row, B_column, "naive_" + std::to_string(i) + " C");
    }

    MLOG("=================== kahan ===================");
    for (size_t i = 8; i <= 32; i *= 2) {
        MATRIX_CHECK_CUDART_ERROR(matrix_multiply_cuda_kahan<float>(dev_A, dev_B, dev_C, A_row, B_column, A_column, i));
        // print_cuda_matrix<float>(dev_C, A_row, B_column, "kahan_" + std::to_string(i) + " C");
    }

    MLOG("=================== shared memory ===================");
    for (size_t i = 8; i <= 32; i *= 2) {
        MATRIX_CHECK_CUDART_ERROR(
            matrix_multiply_cuda_shared_memory<float>(dev_A, dev_B, dev_C, A_row, B_column, A_column, i));
        // print_cuda_matrix<float>(dev_C, A_row, B_column, "shared_memory_" + std::to_string(i) + " C");
    }

    MATRIX_CHECK_CUDART_ERROR(cudaFree(dev_A));
    MATRIX_CHECK_CUDART_ERROR(cudaFree(dev_B));
    MATRIX_CHECK_CUDART_ERROR(cudaFree(dev_C));

    return 0;
}
