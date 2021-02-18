#include "student.hpp"

#include <algorithm>

#include "common.hpp"
#include "chronoGPU.hpp"
#include "chronoCPU.hpp"

namespace IMAC {

	void configureKernelSize(const size_t arraySize, int& threadsByBlock, int& blockNumber) {
		threadsByBlock = MAX_NB_THREADS;
		blockNumber = DEFAULT_NB_BLOCKS;
	}

	__global__
    void reduceKernel(const uint* array, const uint size, uint* partialOut) {
		extern __shared__ uint shared[];

		unsigned int localIdx = threadIdx.x;
		unsigned int globalIdx = localIdx + blockIdx.x * blockDim.x;

		// copy data in shared memory
		shared[localIdx] = array[globalIdx];
		// if(globalIdx < size) {
		// 	shared[localIdx] = array[globalIdx];
		// }else {
		// 	shared[localIdx] = 0; // all other elements to 0
		// }
		__syncthreads();

		// reduction
		for(unsigned int s = 1; s < blockDim.x; s *= 2) {

			// if (localIdx % (2*s) == 0)
			// 	shared[localIdx] = max(shared[localIdx], shared[localIdx + s]);

			const unsigned int StridedIndex = 2 * s * localIdx;
			if(StridedIndex < blockDim.x)
				shared[StridedIndex] = max(shared[StridedIndex], shared[StridedIndex + s]);

			__syncthreads();
		}

		// write result for this block
		if (localIdx == 0) partialOut[blockIdx.x] = shared[0];
	}

	void studentJob(const std::vector<uint>& array, const uint resCPU, const uint nbIterations) {
		// resCPUJust for comparison
		
		uint *dev_array = nullptr;
		uint *dev_partialMax = nullptr;

		HANDLE_ERROR(cudaMalloc( (void**)&dev_array, array.size() * sizeof(uint)));
		HANDLE_ERROR(cudaMemcpy( dev_array, array.data(), array.size() * sizeof(uint), cudaMemcpyHostToDevice));

		// Configure kernel
		int threadsByBlock = MAX_NB_THREADS;
		int blockNumber = DEFAULT_NB_BLOCKS;
		configureKernelSize(array.size(), threadsByBlock, blockNumber);

		verifyDim(blockNumber, threadsByBlock);

		// alloc host and dev partial array
		std::vector<uint> host_partialMax(array.size()/blockNumber, 0);
		HANDLE_ERROR(cudaMalloc((void**) &dev_partialMax, host_partialMax.size() * sizeof(uint) ) );

		const size_t bytesSharedMem = threadsByBlock * sizeof(uint);

		std::cout << "Process on GPU (" << nbIterations << " iterations Avg)" << std::endl;
		std::cout << "Computing on " << blockNumber << " block(s) and "  << threadsByBlock << " thread(s) " <<"- shared mem size = " << bytesSharedMem << std::endl;

		float2 timing = { 0.f, 0.f }; // x: timing GPU, y: timing CPU

		ChronoGPU chrGPU;
		chrGPU.start();
		for (size_t i = 0; i < nbIterations; ++i) { // Average timing on 'loop' iterations
			std::cout << "iteration " << i << std::endl;
			reduceKernel<<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax);
			// std::cout << "Not implemented !" << std::endl;
		}
		chrGPU.stop();
		timing.x = chrGPU.elapsedTime() / nbIterations;

		std::cout << "Test" << std::endl;

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
