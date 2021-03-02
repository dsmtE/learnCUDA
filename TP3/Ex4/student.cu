#include "student.hpp"

#include <algorithm>

#include "common.hpp"
#include "chronoGPU.hpp"
#include "chronoCPU.hpp"

namespace IMAC {

	void configureKernelSize(const size_t arraySize, int& threadsByBlock, int& blockNumber) {
		threadsByBlock = umin(nextPow2<uint>(arraySize), MAX_NB_THREADS);
		blockNumber = arraySize/(2*threadsByBlock)+2;
	}

	__device__ void warpReduce(volatile uint* shared, int i) {
		shared[i] = umax(shared[i], shared[i + 32]);
		shared[i] = umax(shared[i], shared[i + 16]);
		shared[i] = umax(shared[i], shared[i + 16]);
		shared[i] = umax(shared[i], shared[i + 8]);
		shared[i] = umax(shared[i], shared[i + 4]);
		shared[i] = umax(shared[i], shared[i + 2]);
		shared[i] = umax(shared[i], shared[i + 1]);
	}

	__global__
    void reduceKernel(const uint* array, const uint size, uint* partialOut) {
		extern __shared__ uint shared[];

		unsigned int localIdx = threadIdx.x;
		unsigned int globalIdx = localIdx + blockIdx.x * blockDim.x * 2;
	
		// copy data in shared memory and do first level of reduction
		if(globalIdx < size) {
			shared[localIdx] = array[globalIdx];
		
			if(globalIdx + blockDim.x < size)
				shared[localIdx] = umax(shared[localIdx], array[globalIdx + blockDim.x]);
		} else {
			shared[localIdx] = 0; // all other elements to 0
		}

		__syncthreads();

		// reduction
		for (unsigned int s=blockDim.x/2; s>32; s >>= 1) {
			if (localIdx < s) {
				shared[localIdx] = max(shared[localIdx], shared[localIdx + s]);
			}
			__syncthreads();
		}

		if (localIdx < 32) warpReduce(shared, localIdx);

		// write result for this block
		if (localIdx == 0) partialOut[blockIdx.x] = shared[0];
	}

	void studentJob(const std::vector<uint>& array, const uint resCPU, const uint nbIterations) {
		// resCPUJust for comparison
		
		uint *dev_array = nullptr;
		uint *dev_partialMax = nullptr;

		HANDLE_ERROR(cudaMalloc((void**)&dev_array, array.size() * sizeof(uint)));
		HANDLE_ERROR(cudaMemcpy(dev_array, array.data(), array.size() * sizeof(uint), cudaMemcpyHostToDevice));

		// Configure kernel
		int threadsByBlock, blockNumber;
		configureKernelSize(array.size(), threadsByBlock, blockNumber);

		verifyDim(blockNumber, threadsByBlock);

		// alloc host and dev partial array
		std::vector<uint> host_partialMax(blockNumber, 0);
		HANDLE_ERROR(cudaMalloc((void**) &dev_partialMax, host_partialMax.size() * sizeof(uint) ) );

		const size_t bytesSharedMem = threadsByBlock * sizeof(uint);

		std::cout << "Process on GPU (" << nbIterations << " iterations Avg)" << std::endl;
		std::cout << "Computing on " << blockNumber << " block(s) and "  << threadsByBlock << " thread(s) " <<"- shared mem size = " << bytesSharedMem << std::endl;

		float2 timing = { 0.f, 0.f }; // x: timing GPU, y: timing CPU

		ChronoGPU chrGPU;
		chrGPU.start();
		for (size_t i = 0; i < nbIterations; ++i) { // Average timing on 'loop' iterations
			reduceKernel<<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax);
		}
		chrGPU.stop();
		timing.x = chrGPU.elapsedTime() / nbIterations;

		// Retrieve partial result from device to host
		HANDLE_ERROR(cudaMemcpy(host_partialMax.data(), dev_partialMax, host_partialMax.size() * sizeof(uint), cudaMemcpyDeviceToHost));

		// Free array on GPU
		cudaFree(dev_partialMax);
		cudaFree(dev_array);

        // Check for error
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error(cudaGetErrorString(err));
		}
		
		uint result;

		ChronoCPU chrCPU;
		chrCPU.start();
		for (uint i = 0; i < nbIterations; ++i) {
			result = *std::max_element(host_partialMax.begin(), host_partialMax.end());
		}
		chrCPU.stop();

		timing.y = chrCPU.elapsedTime() / nbIterations;
		
        printTiming(timing);
		compare(result, resCPU); // Compare results
	}
}
