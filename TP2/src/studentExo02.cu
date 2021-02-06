/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC {

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c) {
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
// ==================================================

	#define MaxKernelSize 20
	__constant__ float matConv_cu_const[MaxKernelSize * MaxKernelSize];

	__global__ void naiveConv(const uchar4* input, const uint imgWidth, const uint imgHeight, const int matSize, uchar4* output) {
		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		// en inversant les boucles on obtient le même résultat en 0.435808 ms au lieu de 0.060288 ms
		
		if(idx < imgWidth && idy < imgHeight) {

			float3 sum = make_float3(0.f, 0.f, 0.f);

			for (uint j = 0; j < matSize; ++j ){
				for (uint i = 0; i < matSize; ++i ) {
					int dX = idx + i - matSize / 2;
					int dY = idy + j - matSize / 2;

					// Handle borders
					dX = min(max(dX, 0), imgWidth-1);
					dY = min(max(dY, 0), imgHeight-1);

					//dX = abs(abs(dX) - (int)(imgWidth-1)) + (imgWidth-1);
					//dY = abs(abs(dY) - (int)(imgHeight-1)) + (imgHeight-1);

					const int idMat	= j * matSize + i;
					const int idPixel = dY * imgWidth + dX;
					sum.x += (float)input[idPixel].x * matConv_cu_const[idMat];
					sum.y += (float)input[idPixel].y * matConv_cu_const[idMat];
					sum.z += (float)input[idPixel].z * matConv_cu_const[idMat];
				}
			}

			const int idOut = idy * imgWidth + idx;
			output[idOut].x = (uchar)min(max(sum.x, 0.f), 255.f);
			output[idOut].y = (uchar)min(max(sum.y, 0.f), 255.f);
			output[idOut].z = (uchar)min(max(sum.z, 0.f), 255.f);
			output[idOut].w = 255;
		}
	}

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					) {
		ChronoGPU chrGPU;

		// variables
		uchar4* input_cu = nullptr;
		uchar4* output_cu = nullptr;

		//float* matConv_cu = nullptr;
		

		// Allocating
		std::cout << "Allocating:" << std::endl;
		chrGPU.start();
		cudaMalloc((void**)&input_cu, inputImg.size() * sizeof(uchar4));
		cudaMalloc((void**)&output_cu, inputImg.size() * sizeof(uchar4));

		//cudaMalloc((void**)&matConv_cu, mmatSize * sizeof(float));

		chrGPU.stop();
		std::cout 	<< " Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// copy data to GPU
		cudaMemcpy(input_cu, inputImg.data(), inputImg.size() * sizeof(uchar4), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(matConv_cu_const, matConv.data(), matConv.size() * sizeof(float));

		// GPU compute
		std::cout << "Process on GPU " << std::endl;
		chrGPU.start();
		const dim3 dimThreads(32, 32);
		const dim3 dimBlock(imgWidth/dimThreads.x+1, imgHeight/dimThreads.y+1);

		naiveConv<<<dimBlock, dimThreads >>>(input_cu, imgWidth, imgHeight, matSize, output_cu);
		cudaDeviceSynchronize();
		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// copy back to CPU
		cudaMemcpy(output.data(), output_cu, inputImg.size() * sizeof(uchar4), cudaMemcpyDeviceToHost);

		// free memory
		cudaFree(input_cu);
		cudaFree(output_cu);
		cudaFree(matConv_cu_const);
	}
}
