#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>


__host__ __device__ inline float3 pixel_pos(
	int row, 
	int col, 
	float pixel_length, 
	float row_offset, 
	float col_offset) {
	float3 pos = make_float3(row * pixel_length + row_offset, col * pixel_length + col_offset,0);
	return pos;
}

extern "C" void triangle_rasterization(
	float* const d_Data,
	int rows,
	int cols,
	float pixel_length,
	float row_offset,
	float col_offset,
	float3 p0,
	float3 p1,
	float3 p2,
	float3 p3,
	float val
);