#include <iostream>
 
#include "./Test/Configure/common_test.h"
#include "./Test/configure/cuda_config_test.cuh"  // check cuda configure
#include "./Test/Raytracing/raytracing_test.cuh"

#include "./Convolution/Cufft/convolutionFFT2D_test.h"
#include "./Convolution/Rasterization/rasterization_test.h"


int main() {
	/*
	std::cout << "test cuda configure" << std::endl;
	test_cuda_conf();
	*/

    int nFailures = 0;
	
	if (!test_raytracing()) { nFailures++; }
	// if (!common_test::test_file_path()) { nFailures++; }
	// if (!testFastConvolution()){  nFailures++; }
	// if (!test_rasterization()) { nFailures++;  }

	std::cout << "nFailures number: " << nFailures << std::endl;
	system("pause");
	return 0;
}