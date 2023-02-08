// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: matrix common

#ifndef __MATRIX_MULTIPLY_COMMON_H__
#define __MATRIX_MULTIPLY_COMMON_H__

#include <iostream>
#include <random>
#include <vector>

#include "trace_profile.h"

template <typename T>
void get_random_matrix(std::vector<std::vector<T>> &matrix, const T &min, const T &max) {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
    MLOG("matrix: %zu * %zu", matrix.size(), matrix[0].size());
    std::random_device rd;
    std::default_random_engine engine{rd()};
    std::uniform_real_distribution<T> uniform(min, max);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            matrix[i][j] = uniform(engine);
        }
    }
}

template <typename T>
void print_matrix(const std::vector<std::vector<T>> &matrix, const std::string &name = "matrix") {
    MATRIX_TRACE_PROFILE(__FUNCTION__);
    std::cerr << name << ": " << matrix.size() << " * " << matrix[0].size() << std::endl;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            std::cerr << matrix[i][j] << "\t";
        }
        std::cerr << std::endl;
    }
}

#endif  // __MATRIX_MULTIPLY_COMMON_H__
