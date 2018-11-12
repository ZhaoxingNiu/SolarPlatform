#ifndef GEN_KERNEL_H
#define GEN_KERNEL_H

#include "../../Common/global_function.cuh"
#include "../../Common/common_var.h"

void gen_kernel(
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

void gen_kernel_gaussian(
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
void gen_gau_kernel_param(
	float true_distance,
	float &A
);


#endif // !GEN_KERNEL_H
