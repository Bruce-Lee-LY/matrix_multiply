// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: matrix multiply tiling on cpu

#ifndef __MATRIX_MULTIPLY_CPU_TILING_H__
#define __MATRIX_MULTIPLY_CPU_TILING_H__

#include <vector>

// tiling
// C11 = A11 * B11 + A12 * B21
// C12 = A11 * B12 + A12 * B22
// C21 = A21 * B11 + A22 * B21
// C22 = A21 * B12 + A22 * B22

// 8 matrix multiplication and 4 matrix additions, space access locality of A, B and C
template <typename T>
void matrix_multiply_cpu_tiling(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                                std::vector<std::vector<T>> &C, size_t block) {
    MATRIX_TRACE_PROFILE(__FUNCTION__ + std::string("_") + std::to_string(block));
    size_t A_row = A.size();
    size_t A_column = A[0].size();
    size_t B_row = B.size();
    size_t B_column = B[0].size();
    size_t C_row = C.size();
    size_t C_column = C[0].size();
    if (A_column != B_row || A_row != C_row || B_column != C_column) {
        MLOG("input error: A (%zu * %zu) * B (%zu * %zu) != C (%zu * %zu)", A_row, A_column, B_row, B_column, C_row,
             C_column);
        return;
    }

    for (size_t i = 0; i < C_row; ++i) {
        memset(C[i].data(), 0x00, C_column * sizeof(T));
    }

    T tmp = 0.0;
    for (size_t i = 0; i < A_row; i += block) {
        for (size_t k = 0; k < A_column; k += block) {
            for (size_t j = 0; j < B_column; j += block) {
                for (size_t i1 = i; i1 < i + block && i1 < A_row; ++i1) {
                    for (size_t k1 = k; k1 < k + block && k1 < A_column; ++k1) {
                        tmp = A[i1][k1];
                        for (size_t j1 = j; j1 < j + block && j1 < B_column; ++j1) {
                            C[i1][j1] += tmp * B[k1][j1];
                        }
                    }
                }
            }
        }
    }
}

#endif  // __MATRIX_MULTIPLY_CPU_TILING_H__
