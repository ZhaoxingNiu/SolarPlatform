#include "./gen_kernel_test.h"
#include "../../Common/utils.h"

bool testGenFittedKernel(float ori_dis, float true_dis, float angle) {
	bool ret = false;
	genFittedKernel(
		ori_dis, 
		true_dis,
		angle,
		true,
		0.05f,
		0.05f,
		0.1f,
		20.05f,
		20.05f,
		14.0f
	);
	ret = true;
	return ret;
}



bool testLoadKernel() {
	// load the kernel
	std::string kernel_path = "../SimulResult/data/gen_flux/onepoint_angle_0_distance_500.txt";
	std::string check_path = "../SimulResult/data/check/onepoint_angle_0_distance_500.txt";
	
	LoadedConvKernel kernel(201,201,kernel_path);
	kernel.genKernel();
	kernel.saveKernel(check_path);
	
	LoadedConvKernel kernel2(201, 201, check_path);
	kernel2.genKernel();

	checkResultsEps(kernel.h_data, kernel2.h_data, 201*201, 1e-2, 1.0);

	return true;
}