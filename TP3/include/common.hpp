#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef unsigned char uchar;
typedef unsigned int uint;

static void HandleError(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
    	std::stringstream ss;
    	ss << line;
        std::string errMsg(cudaGetErrorString(err));
        errMsg += " (file: " + std::string(file);
        errMsg += " at line: " + ss.str() + ")";
        throw std::runtime_error(errMsg);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template<typename T>
T nextPow2(T x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

namespace IMAC {

    void verifyDim(const uint dimGrid, const uint threadsPerBlock);

    void printTiming(const float2 timing);

    void compare(const uint resGPU, const uint resCPU);
}
