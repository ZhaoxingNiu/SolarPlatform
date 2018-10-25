#include "oblique_parallel.cuh"
#include "../../Common/global_function.cuh"
#include "../Cufft/convolutionFFT2D_common.h"

__global__ void projection_plane_kernel(
	float *d_receiver,   
	float *d_image,          
	float3 rece_pos,        
	float3 rece_u_axis,       
	float3 rece_v_axis,      
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

	if ( r_index.x < rece_size.x && r_index.y < rece_size.y) {
		// reveice position
		float3 r_pos = global_func::index_2_pos(
		    r_index,
			rece_size,
			rece_pixel_len, 
			rece_pos, 
			rece_u_axis, 
			rece_v_axis);
		// get the image plane position 
		float3 i_pos = global_func::matrix_mul_float3(r_pos, d_M) + offset;
		// get the image index 
		int2 i_index = global_func::pos_2_index(
			i_pos,
			image_size,
			image_pixel_len,
			image_pos,
			image_u_axis,
			image_v_axis
		);

		// map the index 
		if (i_index.x >= 0 && i_index.x < image_size.x
			&& i_index.y >= 0 && i_index.y < image_size.y) {
			d_receiver[r_index.x * rece_size.y + r_index.y]
				= d_image[i_index.x * image_size.y + i_index.y];
		}
	}
}

extern "C" void projection_plane_rect(
	float *d_receiver,        // the receiver pixel
	float *d_image,           // the image pixel
	RectangleReceiver *rece,  // receiver
	ProjectionPlane *plane,   // image plane
	float *M,               // the transform matrix, from the metrix
	float3 offset             // the trandform matrix offset
) {
	// get the related prams
	float3 rece_pos = rece->focus_center_;
	float3 rece_u_axis = rece->u_axis_;
	float3 rece_v_axis = rece->v_axis_;
	int2 rece_size = rece->resolution_;
	float rece_pixel_len = rece->pixel_length_;

	float3 image_pos = plane->pos;
	float3 image_u_axis = plane->u_axis;
	float3 image_v_axis = plane->v_axis;
	int2 image_size = { plane->rows, plane->cols };
	float image_pixel_len = plane->pixel_length;

	dim3 threads(32, 32);
	dim3 grid(global_func::iDivUp(rece_size.x, threads.x), global_func::iDivUp(rece_size.y, threads.y));
	
	float *d_M = nullptr;
	global_func::cpu2gpu(d_M, M, 9);

	// for each pixel, calc the mapping function 
	projection_plane_kernel <<< grid, threads >>> (
		d_receiver,
		d_image,
		rece_pos,
		rece_u_axis,
		rece_v_axis,
		rece_size,
		rece_pixel_len,
		image_pos,
		image_u_axis,
		image_v_axis,
		image_size,
		image_pixel_len,
		d_M,
		offset
		);

	checkCudaErrors(cudaFree(d_M));
}