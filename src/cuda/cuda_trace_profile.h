// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:51:47 on Sat, Jun 04, 2022
//
// Description: cuda trace and profile

#ifndef __MATRIX_MULTIPLY_CUDA_TRACE_PROFILE_H__
#define __MATRIX_MULTIPLY_CUDA_TRACE_PROFILE_H__

#include <string>

#include "cuda_common.h"

class CUDATraceProfile {
public:
    CUDATraceProfile(const std::string &name) : m_name(name) {
        MLOG("%s enter", m_name.c_str());
        MATRIX_CHECK_CUDART_ERROR(cudaEventCreate(&m_start));
        MATRIX_CHECK_CUDART_ERROR(cudaEventCreate(&m_stop));

        MATRIX_CHECK_CUDART_ERROR(cudaEventRecord(m_start, NULL));
    }

    ~CUDATraceProfile() {
        MATRIX_CHECK_CUDART_ERROR(cudaEventRecord(m_stop, NULL));
        MATRIX_CHECK_CUDART_ERROR(cudaEventSynchronize(m_stop));
        MATRIX_CHECK_CUDART_ERROR(cudaEventElapsedTime(&m_duration, m_start, m_stop));
        MLOG("%s exit, taken %.3f ms", m_name.c_str(), m_duration);

        MATRIX_CHECK_CUDART_ERROR(cudaEventDestroy(m_start));
        MATRIX_CHECK_CUDART_ERROR(cudaEventDestroy(m_stop));
    }

private:
    CUDATraceProfile(const CUDATraceProfile &) = delete;
    void operator=(const CUDATraceProfile &) = delete;

    const std::string m_name;
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_stop = nullptr;
    float m_duration = 0.0;
};

#ifdef MATRIX_BUILD_DEBUG
#define MATRIX_CUDA_TRACE_PROFILE(name) CUDATraceProfile _tp_##name_(name)
#else
#define MATRIX_CUDA_TRACE_PROFILE(name)
#endif

#endif  // __MATRIX_MULTIPLY_CUDA_TRACE_PROFILE_H__
