#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


typedef unsigned char uchar;
typedef unsigned int uint;

void HandleError(cudaError_t err, const char *file, const int line);

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template<typename T>
inline const T& clamp( const T& val, const T& min, const T& max ) {
    assert( !(max < min) );
    return (val < min) ? min : (max < val) ? max : val;
}

inline std::ostream &operator <<(std::ostream &os, const uchar4 &c) {
    os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    return os; 
}
namespace IMAC {

    void printUsageAndExit(const char *prg);

    void convCPU(const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
                const std::vector<float> &matConv, const uint matSize, std::vector<uchar4> &output);

    void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b);
}
