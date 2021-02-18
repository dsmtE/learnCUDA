#include "common.hpp"

namespace IMAC {

    void verifyDim(const uint dimGrid, const uint threadsPerBlock) {
        cudaDeviceProp prop;
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));

        unsigned long maxGridSize			= prop.maxGridSize[0];
        unsigned long maxThreadsPerBlock	= prop.maxThreadsPerBlock;

        if ( threadsPerBlock > maxThreadsPerBlock ) throw std::runtime_error("Maximum threads per block exceeded");

        if  ( dimGrid > maxGridSize ) throw std::runtime_error("Maximum grid size exceeded");
    }

    void printTiming(const float2 timing) {
        float avg = timing.x + timing.y;
        std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << (timing.x < 1.f ? " us" : " ms") << " on device and ";
        std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << (timing.y < 1.f ? " us" : " ms") << " on host. ("
        <<  ( avg < 1.f ? 1e3f * avg : avg ) << (avg < 1.f ? " us" : " ms") << ")" << std::endl;
    }

    void compare(const uint resGPU, const uint resCPU) {
        if (resGPU == resCPU) {
            std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
        } else {
            std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
        }
    }
}