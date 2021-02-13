#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC {

	#define MaxKernelSize 20
	__constant__ float matConv_cu_const[MaxKernelSize * MaxKernelSize];

	texture<uchar4, 2> tex2DRef;

	__global__ void naiveConv(const uchar4* input, const uint imgWidth, const uint imgHeight, const int matSize, uchar4* output) {
		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		const int idOut = idy * imgWidth + idx;
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

					const uchar4 pixel = tex2D(tex2DRef, dX, dY);

					sum.x += (float)(pixel.x) * matConv_cu_const[idMat];
					sum.y += (float)(pixel.y) * matConv_cu_const[idMat];
					sum.z += (float)(pixel.z) * matConv_cu_const[idMat];
				}
			}

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

		size_t pitch;

		chrGPU.start();

		std::cout << "Allocating:" << std::endl;
		
		cudaMalloc((void**)&output_cu, inputImg.size() * sizeof(uchar4));

		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// copy data to GPU
		cudaMallocPitch((void**)&input_cu, &pitch, imgWidth * sizeof(uchar4), imgHeight);

		cudaMemcpy2D(input_cu, pitch, inputImg.data(),  imgWidth * sizeof(uchar4), imgWidth * sizeof(uchar4), imgHeight, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(matConv_cu_const, matConv.data(), matConv.size() * sizeof(float));
		
		cudaBindTexture2D(NULL, tex2DRef, input_cu, tex2DRef.channelDesc, imgWidth, imgHeight, pitch);

		// GPU compute
		std::cout << "Process:" << std::endl;
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
		cudaUnbindTexture(tex2DRef);
	}
}
