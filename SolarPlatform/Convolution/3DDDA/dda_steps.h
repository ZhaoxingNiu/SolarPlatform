#ifndef DDA_STEPS_H
#define DDA_STEPS_H

#include "../../DataStructure/heliostat.cuh"
#include <vector>

bool set_helios_vertexes_cpu(
	const std::vector<Heliostat *> heliostats,
	const int start_pos,
	const int end_pos,
	float3 *h_helio_vertexs);


#endif // !DDA_STEPS_H

