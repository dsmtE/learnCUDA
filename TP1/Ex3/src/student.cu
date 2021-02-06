/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"
#include <algorithm>

namespace IMAC {

	__global__ void sepiaCUDA(const uchar *const input, uchar *const output, const uint width, const uint height) {

		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		// en inversant les boucles on obtient le même résultat en 0.435808 ms au lieu de 0.060288 ms
		
		if(idx < width && idy < height) {
			const int offset = 3*(idx + idy*width);
			const float red = input[offset + 0];
			const float green = input[offset + 1];
			const float blue = input[offset + 2];

			output[offset + 0] = min(255.f, red * 0.393f + green * 0.769f + blue * 0.189f);
			output[offset + 1] = min(255.f, red * 0.349f + green * 0.686f + blue * 0.168f);
			output[offset + 2] = min(255.f, red * 0.272f + green * 0.534f + blue * 0.131f);
		}
	}

	
	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output) {
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = nullptr;
		uchar *dev_output = nullptr;
		
		// Allocate arrays on device (input and ouput)
		const size_t sizeInBytes = 3 * width * height * sizeof(uchar);
		std::cout << "Allocating input (2 arrays): " << ( ( 2 * sizeInBytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		// allocation
		cudaMalloc((void**)&dev_input, sizeInBytes); // 3 channels
		cudaMalloc((void**)&dev_output, sizeInBytes);

		chrGPU.stop();

		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;
		std::cout << "============================================"	<< std::endl << std::endl;
		std::cout << "Process on GPU " << std::endl;

		cudaMemcpy(dev_input, input.data(), sizeInBytes, cudaMemcpyHostToDevice);
		
		chrGPU.start();
		// Launch a kernel on the GPU
		const dim3 dimThreads(32, 32);
		const dim3 dimBlock(width/dimThreads.x+1, height/dimThreads.y+1);

		sepiaCUDA<<<dimBlock, dimThreads >>>(dev_input, dev_output, width, height);
		cudaDeviceSynchronize();

		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		cudaMemcpy(output.data(), dev_output, sizeInBytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
