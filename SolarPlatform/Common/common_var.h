#ifndef COMMON_VAR_H
#define COMMON_VAR_H

#include <cuda_runtime.h>
#include <string>

using namespace std;

namespace solarenergy {
	//sun ray related default value
	extern float3 sun_dir;
	extern float dni;
	extern float csr;
	extern float num_sunshape_groups;
	extern float num_sunshape_lights_per_group;
	extern int num_sunshape_lights_loop;

	extern float helio_pixel_length;
	extern float receiver_pixel_length;
	extern float image_plane_pixel_length;
	extern float reflected_rate;
	extern float disturb_std;
	 
	//conv related
	extern int2 image_plane_size;
	extern float image_plane_offset;
	extern float total_time;
	extern float total_time2;
	extern int  total_times;

	//default scene file
	extern string scene_filepath;

	// the python script path
	extern string script_filepath;
};

#endif