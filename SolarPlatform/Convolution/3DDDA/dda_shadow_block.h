#ifndef DDA_H
#define DDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <set>
#include "../../SceneProcess/solar_scene.h"



void calc_intersection_3DDDA(
	const std::vector<float3> &vertex,          // the vertex heliostat
	RectGrid &rectgrid,                         // the calc block grid   
	float3 d_dir,                               // the ray direction
	const float3 *h_heliotat_vertex,            // heliostat vertex
	const int const *h_grid_heliostat_match,    // grid and heliostat match relation
	const int const *h_grid_heliostat_index,    // grid index
	std::set<int>& relative_helio_label);       // the heliostat index the shadow

#endif // !DDA_H