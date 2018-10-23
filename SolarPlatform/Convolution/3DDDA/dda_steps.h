#ifndef DDA_STEPS
#define DDA_STEPS

#include "../../DataStructure/heliostat.cuh"
#include <vector>

bool set_helios_vertexes_cpu(
	std::vector<Heliostat *> heliostats,
	const int start_pos,
	const int end_pos,
	float3 *h_helio_vertexs);


#endif // !DDA_STEPS

