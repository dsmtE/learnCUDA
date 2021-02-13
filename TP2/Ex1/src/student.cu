#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC {

	__global__ void naiveConv(const uchar4* input, const uint imgWidth, const uint imgHeight, const float* matConv, const int matSize, uchar4* output) {
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

					const int idMat	= j * matSize + i;
					const int idPixel = dY * imgWidth + dX;
					sum.x += (float)input[idPixel].x * matConv[idMat];
					sum.y += (float)input[idPixel].y * matConv[idMat];
					sum.z += (float)input[idPixel].z * matConv[idMat];
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
		float* matConv_cu = nullptr;
		uchar4* output_cu = nullptr;

		// Allocating
		std::cout << "Allocating:" << std::endl;
		chrGPU.start();
		cudaMalloc((void**)&input_cu, inputImg.size() * sizeof(uchar4));
		cudaMalloc((void**)&matConv_cu, matConv.size() * sizeof(float));
		cudaMalloc((void**)&output_cu, inputImg.size() * sizeof(uchar4));
		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// copy data to GPU
		cudaMemcpy(input_cu, inputImg.data(), inputImg.size() * sizeof(uchar4), cudaMemcpyHostToDevice);
		cudaMemcpy(matConv_cu, matConv.data(), matConv.size() * sizeof(float), cudaMemcpyHostToDevice);

		// GPU compute
		std::cout << "Process:" << std::endl;
		chrGPU.start();
		const dim3 dimThreads(32, 32);
		const dim3 dimBlock(imgWidth/dimThreads.x+1, imgHeight/dimThreads.y+1);

		naiveConv<<<dimBlock, dimThreads >>>(input_cu, imgWidth, imgHeight, matConv_cu, matSize, output_cu);
		cudaDeviceSynchronize();
		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// copy back to CPU
		cudaMemcpy(output.data(), output_cu, inputImg.size() * sizeof(uchar4), cudaMemcpyDeviceToHost);

		// free memory
		cudaFree(input_cu);
		cudaFree(matConv_cu);
		cudaFree(output_cu);
	}
}
