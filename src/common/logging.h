// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: logging

#ifndef __MATRIX_MULTIPLY_LOGGING_H__
#define __MATRIX_MULTIPLY_LOGGING_H__

#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MATRIX_LOG_TAG "MATRIX"
#define MATRIX_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define MLOG(format, ...)                                                                               \
    fprintf(stderr, "[%s %d:%ld %s:%d %s] " format "\n", MATRIX_LOG_TAG, getpid(), syscall(SYS_gettid), \
            MATRIX_LOG_FILE(__FILE__), __LINE__, __FUNCTION__, ##__VA_ARGS__)

#endif  // __MATRIX_MULTIPLY_LOGGING_H__
