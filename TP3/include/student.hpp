#pragma once

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace IMAC {

	typedef unsigned char uchar;
	typedef unsigned int uint;

	const uint MAX_NB_THREADS = 1024;
    const uint DEFAULT_NB_BLOCKS = 32768;

	__global__
    void reduceKernel(const uint* array, const uint size, uint* partialOut);

	void configureKernelSize(const size_t arraySize, int& threadsByBlock, int& blockNumber);

	void studentJob(const std::vector<uint>& array, const uint resCPU, const uint nbIterations);
}
