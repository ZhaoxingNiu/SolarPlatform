#ifndef DDA_INTERFACE_H
#define DDA_INTERFACE_H

#include "../../SceneProcess/PreProcess/scene_instance_process.h"
#include "../Struct/projectionPlane.h"
// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

bool set_helios_vertexes_cpu(
	const std::vector<Heliostat *> &Sheliostats,
	const int start_pos,
	const int end_pos,
	float3 *&h_helio_vertexs);

void dda_interface(
	const SunRay &sunray,                   //  the sun ray
	ProjectionPlane &plane,                 //  the receiver
	const RectangleHelio &recthelio,		//	which heliostat will be traced
	Grid &grid,								//	the grid heliostat belongs to
	const vector<Heliostat *> &heliostats,
	float3 *h_helio_vertexs);	//	all heliostats



#endif // !DDA_INTERFACE_H

