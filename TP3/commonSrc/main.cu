#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <ctime>
#include <climits>
#include <exception>
#include <algorithm>

#include "student.hpp"
#include "chronoCPU.hpp"

namespace IMAC {

    void printUsageAndExit(const char *prg) {
        std::cerr	<< "Usage: " << prg << std::endl
                    << " \t -i <N>: launch <N> iterations (default N = 10)" << std::endl 
                    << " \t -n <N>: size of vector is <N> (default N = 2^23)" << std::endl 
                    << " \t -2n <N>: size of vector is 2^<N> (default N = 23, then size is 2^23)" << std::endl 
                    << std::endl;
        exit(EXIT_FAILURE);
    }

    // Main function
    void main(int argc, char **argv) {
        uint power = 23;
        uint size = 1 << power;
        uint nbIterations = 10;
        // Parse command line
        for ( int i = 1; i < argc; ++i ) {
            if ( !strcmp( argv[i], "-2n" ) )  {
                if ( sscanf( argv[++i], "%u", &power ) != 1 ) {
                    std::cerr << "No power provided..." << std::endl;
                    printUsageAndExit( argv[0] );
                } else {
                    size = 1 << power;
                }
            } else if ( !strcmp( argv[i], "-n" ) )  {
                if ( sscanf( argv[++i], "%u", &size ) != 1 ) {
                    std::cerr << "No size provided..." << std::endl;
                    printUsageAndExit( argv[0] );
                }
            } else if ( !strcmp( argv[i], "-i" ) )  {
                if ( sscanf( argv[++i], "%u", &nbIterations ) != 1 ) {
                    std::cerr << "No number of iterations provided..." << std::endl;
                    printUsageAndExit( argv[0] );
                }
            } else {
                std::cerr << "Unrecognized argument: " << argv[i] << std::endl;
                printUsageAndExit( argv[0] );
            }
        }
        
        std::cout << "Max reduce for an array of size " << size << std::endl;
        
        std::cout << "Allocating array on host, " << ( (size * sizeof(uint)) >> 20 ) << " MB" << std::endl;
        std::vector<uint> array(size, 0);

        std::cout << "Initiliazing array..." << std::endl;
        std::srand(std::time(NULL));

        const uint maxRnd = 79797979;
        for (uint i = 0; i < size; ++i) {
            if(i % 32 == 0)
                array[i] = std::rand() % maxRnd;
            else
                array[i] = 79;
        } 
        if (std::rand() % 2 == 0 ) array[size - 1] = maxRnd + std::rand() % 797979;

        uint resultCPU = 0;

        ChronoCPU chrCPU;
        
        std::cout << "Process on CPU (" << nbIterations << " iterations Avg)" << std::endl;

        chrCPU.start();
        for (size_t i = 0; i < nbIterations; ++i)
            resultCPU = *std::max_element(array.begin(), array.end());
        chrCPU.stop();
        std::cout 	<< "-> Done : " << resultCPU << " (in " << (chrCPU.elapsedTime() / nbIterations) << " ms)" << std::endl;

        std::cout << "============================================"	<< std::endl;

        try{
            studentJob(array, resultCPU, nbIterations);
        } catch(const std::exception &e) {
            throw;
        }
        
    }
}

int main(int argc, char **argv)  {
    try {
        IMAC::main(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}
