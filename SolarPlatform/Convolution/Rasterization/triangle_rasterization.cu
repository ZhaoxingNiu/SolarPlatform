#include "./rasterization_common.h"
#include "./triangle_rasterization.cuh"
#include "../../Common/global_function.cuh"
#include "./reduce_sum.cuh"

//#define SUPER_RESOLUTION_SAMPLE

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
	dim3 grid(global_func::iDivUp(rows, threads.x), global_func::iDivUp(cols, threads.y));

//#ifdef SUPER_RESOLUTION_SAMPLE
//	if (abs(val - 1.0) < Epsilon) {
//		// divide the rect into two triangle 1(p0,p1,p2) and triangle 2(p2,p3,p0)
//		point_in_triangle_super <<<grid, threads >>> (d_Data, p0, p1, p2, rows, cols, pixel_length, row_offset, col_offset, val);
//		point_in_triangle_super <<<grid, threads >>> (d_Data, p2, p3, p0, rows, cols, pixel_length, row_offset, col_offset, val);
//	}
//	else {
//		// divide the rect into two triangle 1(p0,p1,p2) and triangle 2(p2,p3,p0)
//		point_in_triangle <<<grid, threads >>> (d_Data, p0, p1, p2, rows, cols, pixel_length, row_offset, col_offset, val);
//		point_in_triangle <<<grid, threads >>> (d_Data, p2, p3, p0, rows, cols, pixel_length, row_offset, col_offset, val);
//	}
//#else
//#endif // SUPER_RESOLUTION_SAMPLE
	point_in_triangle<<<grid, threads>>>(d_Data, p0, p1, p2, rows, cols, pixel_length, row_offset, col_offset, val);
	point_in_triangle<<<grid, threads>>>(d_Data, p2, p3, p0, rows, cols, pixel_length, row_offset, col_offset, val);

#ifdef MOD_PROJECTION_AREA
	if (abs(val - 1.0) < Epsilon) {
		// divide the rect into two triangle 1(p0,p1,p2) and triangle 2(p2,p3,p0)
		float discreat_area = get_discrete_area(d_Data, rows * cols, pixel_length);
		float true_area = global_func::cal_rect_area(p0,p1,p2,p3);
		float rate = true_area / discreat_area;
		point_mul_val<<<grid, threads>>>(d_Data, rows, cols, rate);
	}
#endif


	
}


extern "C" void sum_rasterization(
	float* const d_Data1,
	float* const d_Data2,
	int rows,
	int cols
) {
	dim3 threads(32, 32);
	dim3 grid(global_func::iDivUp(rows, threads.x), global_func::iDivUp(cols, threads.y));
	sum_array << <grid, threads >> >(d_Data1, d_Data2, rows, cols);
}
