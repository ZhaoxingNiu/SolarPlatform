#include "../../Common/vector_arithmetic.cuh"

#include <cuda.h>

__global__ void point_in_triangle(
	float* const d_Data,
	float3 p1,
	float3 p2,
	float3 p3,
	int rows,
	int cols,
	float pixel_length,
	float row_offset,
	float col_offset,
	float val
) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < rows && y < cols) {
		float3 pos = pixel_pos(x, y, pixel_length, row_offset, col_offset);
		float3 v0 = p3 - p1;
		float3 v1 = p2 - p1;
		float3 v2 = pos - p1;
		float dot00 = dot(v0,v0);
		float dot01 = dot(v0, v1);
		float dot02 = dot(v0, v2);
		float dot11 = dot(v1, v1);
		float dot12 = dot(v1, v2);
		float invDenom = 1 / (dot00*dot11 - dot01*dot01);
		float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
		float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
		if (u >= 0 && v >= 0 && u + v <= 1) {  // ??  < 1? 是否三个边界都属于三角形内部
			d_Data[x * cols + y] = val;
		}
	}
}