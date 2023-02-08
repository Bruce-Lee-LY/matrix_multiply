// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: matrix multiply coppersmith-winograd on cpu

#ifndef __MATRIX_MULTIPLY_CPU_COPPERSMITH_WINOGRAD_H__
#define __MATRIX_MULTIPLY_CPU_COPPERSMITH_WINOGRAD_H__

#include <vector>

#include "cpu_common.h"

template <typename T>
void matrix_multiply_coppersmith_winograd(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
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
    std::vector<std::vector<T>> S1(dim, std::vector<T>(dim)), S2(dim, std::vector<T>(dim)),
        S3(dim, std::vector<T>(dim)), S4(dim, std::vector<T>(dim));
    std::vector<std::vector<T>> T1(dim, std::vector<T>(dim)), T2(dim, std::vector<T>(dim)),
        T3(dim, std::vector<T>(dim)), T4(dim, std::vector<T>(dim));
    std::vector<std::vector<T>> P1(dim, std::vector<T>(dim)), P2(dim, std::vector<T>(dim)),
        P3(dim, std::vector<T>(dim)), P4(dim, std::vector<T>(dim)), P5(dim, std::vector<T>(dim)),
        P6(dim, std::vector<T>(dim)), P7(dim, std::vector<T>(dim));
    std::vector<std::vector<T>> U2(dim, std::vector<T>(dim)), U3(dim, std::vector<T>(dim)),
        U4(dim, std::vector<T>(dim));

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

    // Calculate S, T, P and U
    matrix_add<T>(A21, A22, S1);  // A11 + A22
    matrix_sub<T>(S1, A11, S2);   // S2 = S1 - A11
    matrix_sub<T>(A11, A21, S3);  // S3 = A11 - A21
    matrix_sub<T>(A12, S2, S4);   // S4 = A12 - S2

    matrix_sub<T>(B12, B11, T1);  // T1 = B12 - B11
    matrix_sub<T>(B22, T1, T2);   // T2 = B22 - T1
    matrix_sub<T>(B22, B12, T3);  // T3 = B22 - B12
    matrix_sub<T>(T2, B21, T4);   // T4 = T2 - B21

    matrix_multiply_coppersmith_winograd<T>(A11, B11, P1);  // P1 = A11 * B11
    matrix_multiply_coppersmith_winograd<T>(A12, B21, P2);  // P2 = A12 * B21
    matrix_multiply_coppersmith_winograd<T>(S4, B22, P3);   // P3 = S4 * B22
    matrix_multiply_coppersmith_winograd<T>(A22, T4, P4);   // P4 = A22 * T4
    matrix_multiply_coppersmith_winograd<T>(S1, T1, P5);    // P5 = S1 * T1
    matrix_multiply_coppersmith_winograd<T>(S2, T2, P6);    // P6 = S2 * T2
    matrix_multiply_coppersmith_winograd<T>(S3, T3, P7);    // p7 = S3 * T3

    matrix_add<T>(P1, P2, C11);  // C11 = P1 + P2
    matrix_add<T>(P1, P6, U2);   // U2 = p1 + P6
    matrix_add<T>(U2, P7, U3);   // U3 = U2 + P7
    matrix_add<T>(U2, P5, U4);   // U4 = U2 + P5
    matrix_add<T>(U4, P3, C12);  // C12 = U4 + P3
    matrix_sub<T>(U3, P4, C21);  // C21 = U3 - P4
    matrix_add<T>(U3, P5, C22);  // C22 = U3 + P5

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
// S1 = A21 + A22
// S2 = S1 - A11
// S3 = A11 - A21
// S4 = A12 - S2

// T1 = B12 - B11
// T2 = B22 - T1
// T3 = B22 - B12
// T4 = T2 - B21

// P1 = A11 * B11
// P2 = A12 * B21
// P3 = S4 * B22
// P4 = A22 * T4
// P5 = S1 * T1
// P6 = S2 * T2
// p7 = S3 * T3

// U1 = P1 + P2
// U2 = p1 + P6
// U3 = U2 + P7
// U4 = U2 + P5
// U5 = U4 + P3
// U6 = U3 - P4
// U7 = U3 + P5

// assignment
// C11 = U1
// C12 = U5
// C21 = U6
// C22 = U7

// 7 matrix multiplication and 15 matrix additions, multiplication suitable for higher order dense matrix
template <typename T>
void matrix_multiply_cpu_coppersmith_winograd(const std::vector<std::vector<T>> &A,
                                              const std::vector<std::vector<T>> &B, std::vector<std::vector<T>> &C) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
    matrix_multiply_coppersmith_winograd<T>(A, B, C);
}

#endif  // __MATRIX_MULTIPLY_CPU_COPPERSMITH_WINOGRAD_H__
