#ifndef GEN_KERNEL_H
#define GEN_KERNEL_H

#include "../../Common/global_function.cuh"
#include "../../Common/common_var.h"

void genFittedKernel(
	float true_dis,
	float ori_dis = 500.0f,
	float angel = 0.0f,
	bool flush = true,  // 决定是否更新已存在的文件
	float step_r = 0.05f,
	float grid_len = 0.05f,
	float distance_threshold = 0.1f,
	float rece_width = 10.05f,
	float rece_height = 10.05f,
	float rece_max_r = 7.0f
);

void genKernelGaussian(
	float true_dis,
	float ori_dis = 500.0f,
	float angel = 0.0f,
	bool flush = true,
	float step_r = 0.05f,
	float grid_len = 0.05f,
	float distance_threshold = 0.1f,
	float rece_width = 10.05f,
	float rece_height = 10.05f,
	float rece_max_r = 7.0f
);

// calculate the kernel's total energy
void genGauKernelParam(
	float true_distance,
	float &A
);


#endif // !GEN_KERNEL_H
