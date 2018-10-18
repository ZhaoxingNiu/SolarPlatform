#include "oblique_parallel.cuh"
#include "../../Common/global_function.cuh"
#include "../Cufft/convolutionFFT2D_common.h"



__global__ void projection_plane_kernel(
	float *d_receiver,        // the receiver pixel
	float *d_image,           // the image pixel
	float3 rece_pos,          // the receiver center
	float3 rece_u_axis,       // the receiver u axis, Correspond with x
	float3 rece_v_axis,       // the receiver v axis, corre
	int2 rece_size,
	float rece_pixel_len,
	float3 image_pos,
	float3 image_u_axis,
	float3 image_v_axis,
	int2 image_size,
	float image_pixel_len,
	float *d_M,
	float3 offset
) {
	int2 r_index;
	r_index.x = blockDim.x * blockIdx.x + threadIdx.x;
	r_index.y = blockDim.y * blockIdx.y + threadIdx.y;

	if ( r_index.x < rece_pos.x && r_index.y < rece_pos.y) {
		// reveice position
		float3 r_pos = global_func::index_2_pos(
		    r_index,
			rece_size,
			rece_pixel_len, 
			rece_pos, 
			rece_u_axis, 
			rece_v_axis);
		// get the image plane position 
		float3 i_pos = global_func::matrix_mul_float3(r_pos, d_M);
		// get the image index 
		int2 i_index = global_func::pos_2_index(
			i_pos,
			image_size,
			image_pixel_len,
			image_pos,
			image_u_axis,
			image_v_axis
		);

	}



}