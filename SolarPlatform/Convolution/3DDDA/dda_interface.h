#ifndef DDA_INTERFACE
#define DDA_INTERFACE

#include "../../SceneProcess/PreProcess/scene_instance_process.h"
#include "../Struct/projectionPlane.h"

void dda_interface(
	const SunRay &sunray,                   //  the sun ray
	ProjectionPlane &plane,                 //  the receiver
	const RectangleHelio &recthelio,		//	which heliostat will be traced
	Grid &grid,								//	the grid heliostat belongs to
	const vector<Heliostat *> heliostats);	//	all heliostats



#endif // !DDA_INTERFACE

