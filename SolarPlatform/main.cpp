#include <iostream>
 
#include "./Test/configure/cuda_config_test.cuh"  // check cuda configure


int main() {
	std::cout << "test cuda configure" << std::endl;
	test_cuda_conf();
	system("pause");
	return 0;
}