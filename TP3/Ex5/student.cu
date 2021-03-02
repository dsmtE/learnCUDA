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

	template <unsigned int blockSize>
	__device__ void warpReduce(volatile uint* shared, int i) {
		if (blockSize >= 64) shared[i] = umax(shared[i], shared[i + 32]);
		if (blockSize >= 32) shared[i] = umax(shared[i], shared[i + 16]);
		if (blockSize >= 16) shared[i] = umax(shared[i], shared[i + 8]);
		if (blockSize >= 8) shared[i] = umax(shared[i], shared[i + 4]);
		if (blockSize >= 4) shared[i] = umax(shared[i], shared[i + 3]);
		if (blockSize >= 2) shared[i] = umax(shared[i], shared[i + 2]);
		if (blockSize >= 1) shared[i] = umax(shared[i], shared[i + 1]);
	}

	template <unsigned int blockSize>
    __global__ void reduceKernel(const uint* array, const uint size, uint* partialOut) {
		extern __shared__ uint shared[];

		unsigned int localIdx = threadIdx.x;
		unsigned int globalIdx = localIdx + blockIdx.x * blockDim.x * 2;

		shared[localIdx] = 0; // all elements to 0 by default
		// copy data in shared memory and do first level of reduction
		// remove if(globalIdx + blockDim.x < size) condition because we have only power of 2 size of array
		if(globalIdx < size) shared[localIdx] = umax(array[globalIdx], array[globalIdx + blockDim.x]);

		__syncthreads();

		// unroll reduction
		if (blockSize >= 1024) { if (localIdx < 512) { shared[localIdx] = umax(shared[localIdx], shared[localIdx + 512]); } __syncthreads(); }
		if (blockSize >= 512) { if (localIdx < 256) { shared[localIdx] = umax(shared[localIdx], shared[localIdx + 256]); } __syncthreads(); }
		if (blockSize >= 256) { if (localIdx < 128) { shared[localIdx] = umax(shared[localIdx], shared[localIdx + 128]); } __syncthreads(); }
		if (blockSize >= 128) { if (localIdx < 64) { shared[localIdx] = umax(shared[localIdx], shared[localIdx + 64]); } __syncthreads(); }

		// wrap reduction under 32
		if (localIdx < 32) warpReduce<blockSize>(shared, localIdx);

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

			switch (threadsByBlock) {
				case 1024:
					reduceKernel<1024><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 512:
					reduceKernel<512><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 256:
					reduceKernel<256><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 128:
					reduceKernel<128><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 64:
					reduceKernel<64><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 32:
					reduceKernel<32><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 16:
					reduceKernel<16><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 8:
					reduceKernel<8><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 4:
					reduceKernel<4><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 2:
					reduceKernel<2><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
				case 1:
					reduceKernel<1><<<blockNumber, threadsByBlock, bytesSharedMem>>>(dev_array, array.size(), dev_partialMax); break;
			}
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
