#ifndef OBLIQUE_PARALLEL
#define OBLIQUE_PARALLEL

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include "../../Common/vector_arithmetic.cuh"

// ref. he 2018. An analytical flux density distribution model with a closedform expression for a central receiver system
// get a metrix to get the 
// H = M*R + (b/a)*r
void oblique_proj_matrix(
	float3 r_dir,       // the ray direction
	float3 ori_center,  // the image plane center
	float3 ori_normal,  // the iamge plane normal
	std::vector<float> &M, // the transfunction
	float3 &offset      // the offset
	);


__global__ void projection_plane_kernel(
	float *d_receiver,        // the receiver pixel
	float *d_image,           // the image pixel
	float3 rece_pos,          // the receiver center
	float3 rece_u_axis,       // the receiver u axis, correspond with x
	float3 rece_v_axis,       // the receiver v axis, correpnd with the y
	int2 rece_size,           // the grid num
	float rece_pixel_len,     // the receiver picel length
	float3 image_pos,         // the image plane center, same as rece_pos
	float3 image_u_axis,      // u_axis, correspond with x
	float3 image_v_axis,      // v_axis, correspond with y
	int2 image_size,          // the iamge plane grid number 
	float image_pixel_len,    // the receiver pixel length
	float *d_M,                 // the transform matrix, from the metrix
	float3 offset             // the trandform matrix offset
);


#endif // !OBLIQUE_PARALLEL
