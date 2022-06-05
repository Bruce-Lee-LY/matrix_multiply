// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: trace and profile

#ifndef __MATRIX_MULTIPLY_TRACE_PROFILE_H__
#define __MATRIX_MULTIPLY_TRACE_PROFILE_H__

#include <chrono>
#include <string>

#include "logging.h"

class TraceProfile {
public:
    TraceProfile(const std::string &name) : m_name(name), m_start(std::chrono::steady_clock::now()) {
        MLOG("%s enter", m_name.c_str());
    }

    ~TraceProfile() {
        m_end = std::chrono::steady_clock::now();
        m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start);
        MLOG("%s exit, taken %.3lf ms", m_name.c_str(), m_duration.count());
    }

private:
    TraceProfile(const TraceProfile &) = delete;
    void operator=(const TraceProfile &) = delete;

    const std::string m_name;
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_end;
    std::chrono::duration<double, std::milli> m_duration;
};

#ifdef MATRIX_BUILD_DEBUG
#define MATRIX_TRACE_PROFILE(name) TraceProfile _tp_##name_(name)
#else
#define MATRIX_TRACE_PROFILE(name)
#endif

#endif  // __MATRIX_MULTIPLY_TRACE_PROFILE_H__
