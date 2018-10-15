#include <iostream>

#include "./Test/testCudaConf.cuh"  // check cuda configure

int main() {
	std::cout << "test cuda configure" << std::endl;
	testCudaConf();
	system("pause");
	return 0;
}