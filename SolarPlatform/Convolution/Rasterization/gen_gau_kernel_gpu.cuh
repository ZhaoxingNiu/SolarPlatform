#ifndef GEN_GAU_KERNEL_GPU_H
#define GEN_GAU_KERNEL_GPU_H

#include "../../Common/vector_arithmetic.cuh"
#include "../../Common/global_function.cuh"
#include "./rasterization_common.h"
#include "../../Common/common_var.h"

#include <cuda.h>


void gen_gau_kernel_gpu(
	float* const d_Data,
	int rows,
	int cols,
	float pixel_length,
	float row_offset,
	float col_offset,
	float A,
	float sigma_2);

#endif