#include "dda_steps.h"

// get the vertex
bool set_helios_vertexes_cpu(
	std::vector<Heliostat *> heliostats,
	const int start_pos,
	const int end_pos,
	float3 *h_helio_vertexs){

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