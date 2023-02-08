// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: matrix multiply common on cpu

#ifndef __MATRIX_MULTIPLY_CPU_COMMON_H__
#define __MATRIX_MULTIPLY_CPU_COMMON_H__

#include <vector>

template <typename T>
void matrix_add(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                std::vector<std::vector<T>> &result) {
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
}

template <typename T>
void matrix_sub(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B,
                std::vector<std::vector<T>> &result) {
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
}

#endif  // __MATRIX_MULTIPLY_CPU_COMMON_H__
