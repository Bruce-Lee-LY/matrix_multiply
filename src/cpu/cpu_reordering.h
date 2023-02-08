// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: matrix multiply reordering (ikj, jik, jki, kij, kji) on cpu

#ifndef __MATRIX_MULTIPLY_CPU_REORDERING_H__
#define __MATRIX_MULTIPLY_CPU_REORDERING_H__

#include <vector>

// fast: space access locality of A, B and C
template <typename T>
void matrix_multiply_cpu_reordering_ikj(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                                        std::vector<std::vector<T>> &C) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
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
    for (size_t i = 0; i < A_row; ++i) {
        for (size_t k = 0; k < A_column; ++k) {
            tmp = A[i][k];
            for (size_t j = 0; j < B_column; ++j) {
                C[i][j] += tmp * B[k][j];
            }
        }
    }
}

// middle: space access locality of A and C
template <typename T>
void matrix_multiply_cpu_reordering_jik(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                                        std::vector<std::vector<T>> &C) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
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

    T tmp = 0.0;
    for (size_t j = 0; j < B_column; ++j) {
        for (size_t i = 0; i < A_row; ++i) {
            tmp = 0.0;
            for (size_t k = 0; k < A_column; ++k) {
                tmp += A[i][k] * B[k][j];
            }
            C[i][j] = tmp;
        }
    }
}

// slow: space access locality of B
template <typename T>
void matrix_multiply_cpu_reordering_jki(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                                        std::vector<std::vector<T>> &C) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
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
    for (size_t j = 0; j < B_column; ++j) {
        for (size_t k = 0; k < A_column; ++k) {
            tmp = B[k][j];
            for (size_t i = 0; i < A_row; ++i) {
                C[i][j] += A[i][k] * tmp;
            }
        }
    }
}

// fast: space access locality of A, B and C
template <typename T>
void matrix_multiply_cpu_reordering_kij(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                                        std::vector<std::vector<T>> &C) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
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
    for (size_t k = 0; k < A_column; ++k) {
        for (size_t i = 0; i < A_row; ++i) {
            tmp = A[i][k];
            for (size_t j = 0; j < B_column; ++j) {
                C[i][j] += tmp * B[k][j];
            }
        }
    }
}

// slow: space access locality of B
template <typename T>
void matrix_multiply_cpu_reordering_kji(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                                        std::vector<std::vector<T>> &C) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
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
    for (size_t k = 0; k < A_column; ++k) {
        for (size_t j = 0; j < B_column; ++j) {
            tmp = B[k][j];
            for (size_t i = 0; i < A_row; ++i) {
                C[i][j] += A[i][k] * tmp;
            }
        }
    }
}

#endif  // __MATRIX_MULTIPLY_CPU_REORDERING_H__
