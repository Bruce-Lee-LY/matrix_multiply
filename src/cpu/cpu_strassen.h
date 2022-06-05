// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: matrix multiply strassen on cpu

#ifndef __MATRIX_MULTIPLY_CPU_STRASSEN_H__
#define __MATRIX_MULTIPLY_CPU_STRASSEN_H__

#include <vector>

#include "cpu_common.h"

template <typename T>
void matrix_multiply_strassen(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                              std::vector<std::vector<T>> &C) {
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

    if (A_row == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    size_t dim = A_row / 2;

    std::vector<std::vector<T>> A11(dim, std::vector<T>(dim)), A12(dim, std::vector<T>(dim)),
        A21(dim, std::vector<T>(dim)), A22(dim, std::vector<T>(dim));
    std::vector<std::vector<T>> B11(dim, std::vector<T>(dim)), B12(dim, std::vector<T>(dim)),
        B21(dim, std::vector<T>(dim)), B22(dim, std::vector<T>(dim));
    std::vector<std::vector<T>> C11(dim, std::vector<T>(dim)), C12(dim, std::vector<T>(dim)),
        C21(dim, std::vector<T>(dim)), C22(dim, std::vector<T>(dim));
    std::vector<std::vector<T>> P1(dim, std::vector<T>(dim)), P2(dim, std::vector<T>(dim)),
        P3(dim, std::vector<T>(dim)), P4(dim, std::vector<T>(dim)), P5(dim, std::vector<T>(dim)),
        P6(dim, std::vector<T>(dim)), P7(dim, std::vector<T>(dim));
    std::vector<std::vector<T>> A_result(dim, std::vector<T>(dim)), B_result(dim, std::vector<T>(dim));

    // divide original matrix into 4 sub-matrix
    // C11 = A11 * B11 + A12 * B21
    // C12 = A11 * B12 + A12 * B22
    // C21 = A21 * B11 + A22 * B21
    // C22 = A21 * B12 + A22 * B22
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + dim];
            A21[i][j] = A[i + dim][j];
            A22[i][j] = A[i + dim][j + dim];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + dim];
            B21[i][j] = B[i + dim][j];
            B22[i][j] = B[i + dim][j + dim];
        }
    }

    // Calculate P1 to P7
    matrix_add<T>(A11, A22, A_result);                    // A11 + A22
    matrix_add<T>(B11, B22, B_result);                    // B11 + B22
    matrix_multiply_strassen<T>(A_result, B_result, P1);  // P1 = (A11 + A22) * (B11 + B22)

    matrix_add<T>(A21, A22, A_result);               // A21 + A22
    matrix_multiply_strassen<T>(A_result, B11, P2);  // P2 = (A21 + A22) * B11

    matrix_sub<T>(B12, B22, B_result);               // B12 - B22
    matrix_multiply_strassen<T>(A11, B_result, P3);  // P3 = A11 * (B12 - B22)

    matrix_sub<T>(B21, B11, B_result);               // B21 - B11
    matrix_multiply_strassen<T>(A22, B_result, P4);  // P4 = A22 * (B21 - B11)

    matrix_add<T>(A11, A12, A_result);               // A11 + A12
    matrix_multiply_strassen<T>(A_result, B22, P5);  // P5 = (A11 + A12) * B22

    matrix_sub<T>(A21, A11, A_result);                    // A21 - A11
    matrix_add<T>(B11, B12, B_result);                    // B11 + B12
    matrix_multiply_strassen<T>(A_result, B_result, P6);  // P6 = (A21 - A11) * (B11 + B12)

    matrix_sub<T>(A12, A22, A_result);                    // A12 - A22
    matrix_add<T>(B21, B22, B_result);                    // B21 + B22
    matrix_multiply_strassen<T>(A_result, B_result, P7);  // p7 = (A12 - A22) * (B21 + B22)

    // calculate C11, C12, C21 and C22:
    matrix_add<T>(P1, P4, A_result);        // P1 + P4
    matrix_add<T>(A_result, P7, B_result);  // P1 + P4 + P7
    matrix_sub<T>(B_result, P5, C11);       // C11 = P1 + P4 - P5 + P7

    matrix_add<T>(P3, P5, C12);  // C12 = P3 + P5

    matrix_add<T>(P2, P4, C21);  // C21 = P2 + P4

    matrix_add<T>(P1, P3, A_result);        // P1 + P3
    matrix_add<T>(A_result, P6, B_result);  // P1 + P3 + P6
    matrix_sub<T>(B_result, P2, C22);       // C22 = P1 + P3 - P2 + P6

    // put results together
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + dim] = C12[i][j];
            C[i + dim][j] = C21[i][j];
            C[i + dim][j + dim] = C22[i][j];
        }
    }
}

// tiling
// C11 = A11 * B11 + A12 * B21
// C12 = A11 * B12 + A12 * B22
// C21 = A21 * B11 + A22 * B21
// C22 = A21 * B12 + A22 * B22

// intermediate matrix
// P1 = (A11 + A22) * (B11 + B22)
// P2 = (A21 + A22) * B11
// P3 = A11 * (B12 - B22)
// P4 = A22 * (B21 - B11)
// P5 = (A11 + A12) * B22
// P6 = (A21 - A11) * (B11 + B12)
// p7 = (A12 - A22) * (B21 + B22)

// recalculate
// C11 = P1 + P4 - P5 + P7
// C12 = P3 + P5
// C21 = P2 + P4
// C22 = P1 + P3 - P2 + P6

// 7 matrix multiplication and 18 matrix additions, multiplication suitable for higher order dense matrix
template <typename T>
void matrix_multiply_cpu_strassen(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                                  std::vector<std::vector<T>> &C) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
    matrix_multiply_strassen<T>(A, B, C);
}

#endif  // __MATRIX_MULTIPLY_CPU_STRASSEN_H__
