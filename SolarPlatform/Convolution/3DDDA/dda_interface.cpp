#include "./dda_interface.h"
#include "./dda_steps.h"


void dda_interface(
	const SunRay &sunray,                   //  the sun ray
	ProjectionPlane &plane,                 //  the receiver
	const RectangleHelio &recthelio,		//	which heliostat will be traced
	Grid &grid,								//	the grid heliostat belongs to
	const vector<Heliostat *> heliostats)	//	all heliostats
{
	// step 1. get tht ptr vertex
	std::vector<float3> vertex;
	for (int i = 0; i < 4; ++i) {
		vertex.push_back(recthelio.vertex_[i]);
	}

	// step 2. get the all vertex of the heliostat
	float3 *h_helio_vertexs = nullptr;












	delete[] h_helio_vertexs;
	h_helio_vertexs = nullptr;
} 