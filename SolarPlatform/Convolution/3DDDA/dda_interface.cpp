#include "./dda_interface.h"
#include "./dda_steps.h"
#include "./dda_shadow_block.h"


// get the vertex
bool set_helios_vertexes_cpu(
	const std::vector<Heliostat *> &heliostats,
	const int start_pos,
	const int end_pos,
	float3 *&h_helio_vertexs) {
	int size = end_pos - start_pos;
	h_helio_vertexs = new float3[size * 3];

	for (int i = start_pos; i < end_pos; ++i) {
		int offset = i - start_pos;
		// only save the 0 1 3 vertexs
		float3 v0, v1, v3;
		heliostats[i]->Cget_vertex(v0, v1, v3);
		h_helio_vertexs[3 * offset] = v0;
		h_helio_vertexs[3 * offset + 1] = v1;
		h_helio_vertexs[3 * offset + 2] = v3;
	}
	return true;
}



void dda_interface(
	const SunRay &sunray,                   //  the sun ray
	ProjectionPlane &plane,                 //  the receiver
	const RectangleHelio &recthelio,		//	which heliostat will be traced
	Grid &grid,								//	the grid heliostat belongs to
	const vector<Heliostat *> &heliostats,
	float3 *h_helio_vertexs)	//	all heliostats
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// step 2. get tht ptr vertex
	std::vector<float3> vertex;
	for (int i = 0; i < 4; ++i) {
		vertex.push_back(recthelio.vertex_[i]);
	}

	// step 3 dda get the relative heliostat
	// step 3.1 get dir
	float3 in_dir = sunray.sun_dir_;
	in_dir = -in_dir;
	float3 out_dir = reflect(sunray.sun_dir_, recthelio.normal_);   // reflect light
	out_dir = normalize(out_dir);

	std::set<int> shadow_helio_label;
	std::set<int> block_helio_label;
	// now just complete the rectgrid
	if (grid.type_ != 0) {
		std::cerr << "now the code can only process the rectgrid" << std::endl;
		return;
	}

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

	// step 3.3 get the block heliostats
	calc_intersection_3DDDA(
		vertex,
		*rectgrid,
		out_dir,
		h_helio_vertexs,
		rectgrid->h_grid_helio_match_,
		rectgrid->h_grid_helio_index_,
		block_helio_label
	);

	sdkStopTimer(&hTimer);
	double gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;
	printf("3D DDA cost time: (%f ms)\n", gpuTime);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// step 4 process the vertex
	// step 4.1 calculate the contour on the image plane
	std::vector<float3> vec_project;
	std::vector<std::vector<float3>> vecvec_shadow_block;
	for(auto ver: vertex) {
		float3 p;
		plane.ray_intersect_pos2(ver, out_dir, p);
		vec_project.push_back(p);
	}

	// step 4.2 calculate the shadow contour on the image plane
	for (auto index: shadow_helio_label) {
		// get the contour on the heliostat
		float3 v0, v1, v2, v3;
		float3 v0_p, v1_p, v2_p, v3_p;
		float3 v0_i, v1_i, v2_i, v3_i;
		heliostats[index]->Cget_all_vertex(v0,v1,v2,v3);
		recthelio.Cray_intersect(v0, in_dir, v0_p);
		recthelio.Cray_intersect(v1, in_dir, v1_p);
		recthelio.Cray_intersect(v2, in_dir, v2_p);
		recthelio.Cray_intersect(v3, in_dir, v3_p);

		// get the contour on the image plane
		plane.ray_intersect_pos2(v0_p, out_dir, v0_i);
		plane.ray_intersect_pos2(v1_p, out_dir, v1_i);
		plane.ray_intersect_pos2(v2_p, out_dir, v2_i);
		plane.ray_intersect_pos2(v3_p, out_dir, v3_i);

	    std::vector<float3> vec_shadow_block = {v0_i, v1_i, v2_i, v3_i};
		vecvec_shadow_block.push_back(vec_shadow_block);
	}

	// step 4.3 calculate the block coutour on the image plane
	for (auto index: block_helio_label) {
		float3 v0, v1, v2, v3;
		float3 v0_i, v1_i, v2_i, v3_i;
		heliostats[index]->Cget_all_vertex(v0, v1, v2, v3);

		// get the contour on the image plane
		plane.ray_intersect_pos2(v0, out_dir, v0_i);
		plane.ray_intersect_pos2(v1, out_dir, v1_i);
		plane.ray_intersect_pos2(v2, out_dir, v2_i);
		plane.ray_intersect_pos2(v3, out_dir, v3_i);

		std::vector<float3> vec_shadow_block = { v0_i, v1_i, v2_i, v3_i };
		vecvec_shadow_block.push_back(vec_shadow_block);
	}

	// step 5. rasterization the image plane
	plane.projection(vec_project);
	plane.shadow_block(vecvec_shadow_block);
	
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;
	printf("rasterization cost time: (%f ms)\n", gpuTime);
} 