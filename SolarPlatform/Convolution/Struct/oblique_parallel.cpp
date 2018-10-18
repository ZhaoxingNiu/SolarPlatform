#include "oblique_parallel.cuh"

void oblique_proj_matrix(
	float3 r_dir, 
	float3 ori_center, 
	float3 ori_normal, 
	std::vector<float> &M,
	float3 &offset) {
	M.clear();
	M.resize(9);
	float3 n = ori_normal;
	float3 r = r_dir;
	float3 o = ori_center;
	float a = dot(n,r);
	float b = dot(n,o);
	M[0] = 1 - n.x*r.x / a;
	M[1] =   - n.y*r.x / a;
	M[2] =   - n.z*r.x / a;
	M[3] =   - n.x*r.y / a;
	M[4] = 1 - n.y*r.y / a;
	M[5] =   - n.z*r.y / a;
	M[6] =   - n.x*r.z / a;
	M[7] =   - n.y*r.z / a;
	M[8] = 1 - n.z*r.z / a;
	offset = b / a * r;
}