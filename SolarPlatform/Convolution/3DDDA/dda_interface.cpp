#include "./dda_interface.h"
#include "./dda_steps.h"
#include "./dda_shadow_block.h"


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
	int start_pos = grid.start_helio_pos_;
	int end_pos = start_pos + grid.num_helios_;
	set_helios_vertexes_cpu(heliostats, start_pos, end_pos, h_helio_vertexs);

	// step 3 dda get the relative heliostat
	// step 3.1 get dir
	float3 in_dir = sunray.sun_dir_;
	in_dir = -in_dir;
	float3 out_dir = reflect(sunray.sun_dir_, recthelio.normal_);   // reflect light
	out_dir = -out_dir;

	std::set<int> shadow_helio_label;
	std::set<int> block_helio_label;
	// now just complete the rectgrid
	switch (grid.type_) {
	case 0: {
		RectGrid *rectgrid = dynamic_cast<RectGrid *> (&grid);
		// step 3.2 get the shadow heliostats
		calc_intersection_3DDDA(
			vertex,
			*rectgrid,
			in_dir,
			h_helio_vertexs,
			rectgrid->h_grid_helio_match_,
			rectgrid->h_grid_helio_index_,
			shadow_helio_label
		);

		// step 3.3 get the shadow heliostats
		calc_intersection_3DDDA(
			vertex,
			*rectgrid,
			in_dir,
			h_helio_vertexs,
			rectgrid->h_grid_helio_match_,
			rectgrid->h_grid_helio_index_,
			block_helio_label
		);

		// step 4 process the vertex
		



		break;
	}
	default: break;
	}

	delete[] h_helio_vertexs;
	h_helio_vertexs = nullptr;
} 