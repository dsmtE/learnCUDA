/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{

	const int threadsByBlock = 512;

	__global__ void sumArraysCUDA(const int *const dev_a, const int *const dev_b, int *const dev_res, const int size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
    	if (idx < size) {
			dev_res[idx] = dev_a[idx] + dev_b[idx];
		}
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = nullptr;
		int *dev_b = nullptr;
		int *dev_res = nullptr;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		// allocation
		cudaMalloc((void**)&dev_a, size * sizeof(int));
		cudaMalloc((void**)&dev_b, size * sizeof(int));
		cudaMalloc((void**)&dev_res, size * sizeof(int));

		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

		std::cout << "============================================"	<< std::endl;
		std::cout << "Addition on GPU "	<< std::endl;
		chrGPU.start();

		// Launch a kernel on the GPU with one thread for each element.
    	// 2 is number of computational blocks and 256 is a number of threads in a block
		const int blockNumber = size/threadsByBlock+1;
		sumArraysCUDA<<<blockNumber, threadsByBlock >>>(dev_a, dev_b, dev_res, size);
		
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaDeviceSynchronize();

		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Copy output vector from GPU buffer to host memory. (device to host)
		cudaMemcpy(res, dev_res, size * sizeof(int), cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
	}
}

