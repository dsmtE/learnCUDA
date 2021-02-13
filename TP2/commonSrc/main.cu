#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include <algorithm>

#include "student.hpp"
#include "chronoCPU.hpp"
#include "lodepng.h"
#include "conv_utils.hpp"

namespace IMAC {

	void main(int argc, char **argv)  {
		char fileName[2048];
		uint convType;
		// Parse command line
		if (argc != 5) {
			std::cerr << "Wrong number of argument" << std::endl;
			printUsageAndExit(argv[0]);
		}

		for (int i = 1; i < argc; ++i) {
			if (!strcmp(argv[i], "-f")) {
				if (sscanf(argv[++i], "%s", fileName) != 1) {
					std::cerr << "No file provided after -f" << std::endl;
					printUsageAndExit(argv[0]);
				}
			} else if(!strcmp(argv[i], "-c")) {
				if (sscanf(argv[++i], "%u", &convType) != 1) {
					std::cerr << "No index after -c" << std::endl;
					printUsageAndExit(argv[0]);
				}
			} else {
				printUsageAndExit(argv[0]);
			}
		}
		
		// Get input image
		std::vector<uchar> inputUchar;
		uint imgWidth;
		uint imgHeight;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(inputUchar, imgWidth, imgHeight, fileName, LCT_RGBA);
		if (error) {
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		// Convert to uchar4 for exercise convenience
		std::vector<uchar4> input;
		input.resize(inputUchar.size() / 4);
		for (uint i = 0; i < input.size(); ++i) {
			const uint id = 4 * i;
			input[i].x = inputUchar[id];
			input[i].y = inputUchar[id + 1];
			input[i].z = inputUchar[id + 2];
			input[i].w = inputUchar[id + 3];
		}
		inputUchar.clear();
		std::cout << "Image has " << imgWidth << " x " << imgHeight << " pixels (RGBA)" << std::endl;

		// Init convolution matrix
		std::vector<float> matConv;
		uint matSize;
		initConvolutionMatrix(convType, matConv, matSize);

		// Create 2 output images
		std::vector<uchar4> outputCPU(imgWidth * imgHeight);
		std::vector<uchar4> outputGPU(imgWidth * imgHeight);

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string convStr = convertConvTypeToString(convType);
		std::string outputCPUName = name + convStr + "_CPU" + ext;
		std::string outputGPUName = name + convStr + "_GPU" + ext;

		std::cout << std::endl << "Process on CPU" << std::endl;
		std::cout << "============================================"	<< std::endl << std::endl;
		
		convCPU(input, imgWidth, imgHeight, matConv, matSize, outputCPU);
		
		std::cout << "Save image as: " << outputCPUName << std::endl;
		error = lodepng::encode(outputCPUName, reinterpret_cast<uchar *>(outputCPU.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error) { throw std::runtime_error("Error loadpng::encode: " + std::string(lodepng_error_text(error))); }

		std::cout << std::endl << "Process on GPU"<< std::endl;
		std::cout << "============================================"	<< std::endl << std::endl;
		studentJob(input, imgWidth, imgHeight, matConv, matSize, outputCPU, outputGPU);

		std::cout << "Save image as: " << outputGPUName << std::endl;
		error = lodepng::encode(outputGPUName, reinterpret_cast<uchar *>(outputGPU.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error) { throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error))); }
		
		std::cout << "============================================"	<< std::endl << std::endl;

		compareImages(outputCPU, outputGPU);
	}
}

int main(int argc, char **argv) {
	try{
		IMAC::main(argc, argv);
	} catch (const std::exception &e){
		std::cerr << e.what() << std::endl;
	}
}
