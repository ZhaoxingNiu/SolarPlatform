#include "oblique_parallel.cuh"

__global__ void projection_plane(
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
	float *M,
	float3 offset
) {

}