#include "common.hpp"

#include "chronoCPU.hpp"
#include "conv_utils.hpp"

void HandleError(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
    	std::stringstream ss;
    	ss << line;
        std::string errMsg(cudaGetErrorString(err));
        errMsg += " (file: " + std::string(file);
        errMsg += " at line: " + ss.str() + ")";
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

namespace IMAC {
	void printUsageAndExit(const char *prg) {
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name (required)" << std::endl
					<< " \t -c <C>: <C> convolution type (required)" << std::endl 
					<< " \t --- " << BUMP_3x3 << " = Bump 3x3" << std::endl
					<< " \t --- " << SHARPEN_5x5 << " = Sharpen 5x5" << std::endl
					<< " \t --- " << EDGE_DETECTION_7x7 << " = Edge detection 7x7" << std::endl
					<< " \t --- " << MOTION_BLUR_15x15 << " = Motion Blur 15x15" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

    void convCPU(const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
                const std::vector<float> &matConv, const uint matSize, std::vector<uchar4> &output) {
		ChronoCPU chrCPU;
		chrCPU.start();
		for ( uint y = 0; y < imgHeight; ++y ) {
			for ( uint x = 0; x < imgWidth; ++x ) {
				float3 sum = make_float3(0.f, 0.f, 0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j )  {
					for ( uint i = 0; i < matSize; ++i )  {
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;

						// Handle borders
                        dX = clamp<int>(dX, 0, imgWidth-1);
                        dY = clamp<int>(dY, 0, imgHeight-1);

						const int idMat = j * matSize + i;
						const int idPixel = dY * imgWidth + dX;
						sum.x += (float)input[idPixel].x * matConv[idMat];
						sum.y += (float)input[idPixel].y * matConv[idMat];
						sum.z += (float)input[idPixel].z * matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)clamp( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clamp( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clamp( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b) {
		bool error = false;
		if (a.size() != b.size()) {
			std::cout << "Size is different !" << std::endl;
			error = true;
		} else {
			for (uint i = 0; i < a.size(); ++i) {
				// Floating precision can cause small difference between host and device
				if (std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2) {
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error) {
			std::cout << " -> You failed, retry!" << std::endl;
		} else {
			std::cout << " -> Well done!" << std::endl;
		}
	}
}