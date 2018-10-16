#include <iostream>
 
#include "./Test/configure/cuda_config_test.cuh"  // check cuda configure
#include "./Convolution/Cufft/convolutionFFT2D_test.h"


int main() {
	/*
	std::cout << "test cuda configure" << std::endl;
	test_cuda_conf();
	*/

    int nFailures = 0;
	if (!testFastConvolution()){  
		nFailures++; 
	}
	std::cout << "nFailures number: " << nFailures << std::endl;
	system("pause");
	return 0;
}