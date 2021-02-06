#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include <algorithm>

#include "common.hpp"
#include "chronoCPU.hpp"
#include "chronoGPU.hpp"

namespace IMAC {

	void fillRandomMatrix(float* m, const size_t size) {
		for (size_t i = 0; i < size; i++) {
			m[i] = rand()%100;
		}
	}

	void matSumCPU(const float* m1, const float* m2, float* res, const size_t size) {
		for (size_t i = 0; i < size; i++) {
			res[i] = m1[i] + m2[i];
		}
	}

	__global__ void matSumGPU(const float* a, const float* b, const size_t cols, const size_t rows, float* res){
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int index = (idy * cols + idx);
		if (idy < rows && idx < cols){
			res[index] = a[index] + b[index];
		}
	}

	bool compare(const size_t row, const size_t col, const float* a, const float* b) {
		for (auto j = 0; j < row; ++j) {
			for (auto i = 0; i < col; ++i) {
				if (std::abs(a[i + col*j] - b[i + col*j]) > 1e-5) {
					std::cout << "Error at index [" << i << ", " << j << "] : a = " << a[i + col*j] << " - b = " << b[i + col*j] << std::endl;
					return false; 
				}
			}
		}
		return true;
	}

	void GPUCompute(const float* a, const float* b, const size_t rows, const size_t cols, float* res) {
		ChronoGPU chrGPU;

		float* dev_a = NULL;
		float* dev_b = NULL;
		float* dev_res = NULL;

		const size_t bytes = rows*cols*sizeof(float);

		cudaMalloc((void**)&dev_a, bytes);
		cudaMalloc((void**)&dev_b, bytes);
		cudaMalloc((void**)&dev_res, bytes);

		//The error lays here
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);
			
		const dim3 dimThreads(32, 32);
		const dim3 dimBlock(cols/dimThreads.x+1, rows/dimThreads.y+1);
		chrGPU.start();
		matSumGPU<<<dimBlock, dimThreads>>>(dev_a, dev_b, cols, rows, dev_res);
		chrGPU.stop();
		std::cout << "\n -> GPU Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		cudaDeviceSynchronize();

		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);
		
		cudaFree(dev_res);
		cudaFree(dev_b);
		cudaFree(dev_a);
	}

	void main(int argc, char **argv) {	
		ChronoCPU chrCPU;

		size_t rows = 100;
		size_t cols = 120;

		float* a = new float[rows*cols];
		float* b = new float[rows*cols];
		float* resCPU = new float[rows*cols];
		float* resGPU = new float[rows*cols];

		fillRandomMatrix(a, rows*cols);
		fillRandomMatrix(b, rows*cols);

		chrCPU.start();
		matSumCPU(a, b, resCPU, rows*cols);
		chrCPU.stop();
		std::cout 	<< " -> CPU Done : " << chrCPU.elapsedTime() << " ms" << std::endl;

		GPUCompute(a, b, rows, cols, resGPU);

		if (compare(rows, cols, resCPU, resGPU)) {
			std::cout << " -> Well done!" << std::endl;
		} else {
			std::cout << " -> You failed, retry!" << std::endl;
		}

		// Free memory
		delete[] a;
		delete[] b;
		delete[] resCPU;
		delete[] resGPU;
	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
