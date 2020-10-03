//
// Created by linghu8812 on 2020/10/3.
//

#ifndef SERESNEXT_TRT_COMMON_H
#define SERESNEXT_TRT_COMMON_H

#include <cuda_runtime_api.h>
#include <numeric>
#include "logging.h"

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

Logger gLogger{Logger::Severity::kINFO};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
}

#endif //SERESNEXT_TRT_COMMON_H
