#include "./gen_gau_kernel_gpu.cuh"

__host__ __device__ inline float my_gaussian_gpu(float x_pos, float y_pos, float A, float sigma_2) {
	return A / 2.0 / MATH_PI / sigma_2 * expf(-(x_pos * x_pos + y_pos * y_pos) / 2 / sigma_2);
}

__global__ void gen_gau_kernel_gpu_kernel(
	float* const d_Data,
	int rows,
	int cols,
	float pixel_length,
	float row_offset,
	float col_offset,
	float A,
	float sigma_2) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < rows && y < cols) {
		float3 pos = pixel_pos(x, y, pixel_length, row_offset, col_offset);
		d_Data[x * cols + y] = my_gaussian_gpu(pos.x, pos.y, A, sigma_2);
	}
}


void gen_gau_kernel_gpu(
	float* const d_Data,
	int rows,
	int cols,
	float pixel_length,
	float row_offset,
	float col_offset,
	float A,
	float sigma_2) {
	dim3 threads(32, 32);
	dim3 grid(global_func::iDivUp(rows, threads.x), global_func::iDivUp(cols, threads.y));

	gen_gau_kernel_gpu_kernel << <grid, threads >> > (d_Data, rows, cols,
		pixel_length, row_offset, col_offset, A, sigma_2);



}