#ifndef STEPS_FOR_RAYTRACING_H
#define STEPS_FOR_RAYTRACING_H

#include "../SceneProcess/solar_scene.h"

// float3 *d_microhelio_centers
// float3 *d_microhelio_normals
// microhelio_num
bool set_microhelio_centers(const RectangleHelio &recthelio, float3 *&d_microhelio_centers, float3 *&d_microhelio_normals, size_t &size);

// float3 *d_helio_vertexs
//	- start_pos:	start position of heliostats array
//	- end_pos:		the position after the end of heliostats array
bool set_helios_vertexes(vector<Heliostat *> heliostats, const int start_pos, const int end_pos,
							float3 *&d_helio_vertexs);

// int *d_microhelio_groups
bool set_microhelio_groups(int *&d_microhelio_groups, const int num_group, const size_t &size);

#endif // !STEPS_FOR_RAYTRACING_H