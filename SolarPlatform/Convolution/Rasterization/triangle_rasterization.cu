#include "./rasterization_common.h"
#include "./triangle_rasterization.cuh"
#include "../Cufft/convolutionFFT2D_common.h"


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
) {
	dim3 threads(32, 32);
	dim3 grid(iDivUp(rows, threads.x), iDivUp(cols, threads.y));

	// divide the rect into two triangle 1(p0,p1,p2) and triangle 2(p2,p3,p0)
	point_in_triangle <<<grid, threads >>> (d_Data, p0, p1, p2, rows, cols, pixel_length, row_offset, col_offset, val);
	point_in_triangle <<<grid, threads >>> (d_Data, p2, p3, p0, rows, cols, pixel_length, row_offset, col_offset, val);
}
