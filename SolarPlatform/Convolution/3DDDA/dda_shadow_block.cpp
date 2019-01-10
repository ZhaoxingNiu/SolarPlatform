#include "dda_shadow_block.h"


inline float cal_t_max(const float &dir, const float &interval, const int &current_index, const float &current_pos)
{
	if (dir >= 0)
		return float(current_index + 1)*interval - current_pos;
	else
		return current_pos - float(current_index)*interval;
}

template <typename T>
inline T abs_divide(const T &denominator, const T &numerator)
{
	if (numerator <= Epsilon && numerator >= -Epsilon)
		return T(INT_MAX);
	return abs(denominator / numerator);
}

// calculate the intersection
// insert the heliostat index into the set
int intersect_helio(
	const float3 &orig, 
	const float3 &dir, 
	const int grid_address,
	const float3 *h_heliotat_vertex,
	const int const *h_grid_heliostat_match,
	const int const *h_grid_heliostat_index,
	std::set<int>& relative_helio_label) {

	int ret = 0;
	// get all the heliostat in the grid
	for (unsigned int i = h_grid_heliostat_index[grid_address];
		i < h_grid_heliostat_index[grid_address + 1]; ++i) {
		unsigned int heliostat_index = 3 * h_grid_heliostat_match[i];
		float u, v, t;
		bool intersect = global_func::rayParallelogramIntersect(
			orig, dir, 
			h_heliotat_vertex[heliostat_index+0], 
			h_heliotat_vertex[heliostat_index+1],
			h_heliotat_vertex[heliostat_index+2],
			t, u, v
		);
		if (intersect) {
			relative_helio_label.insert(h_grid_heliostat_match[i]);
			++ret;
		}
	}
	return ret;
}

void calc_intersection_3DDDA(
	const std::vector<float3> &vertex,          // the vertex heliostat
	RectGrid &rectgrid,                         // the calc block grid   
	float3 d_dir,                               // the ray direction
	const float3 *h_heliotat_vertex,            // heliostat vertex
	const int const *h_grid_heliostat_match,    // grid and heliostat match relation
	const int const *h_grid_heliostat_index,    // grid index
	std::set<int>& relative_helio_label        // the heliostat index the shadow
){
	size_t vertex_size = vertex.size();
	for (size_t v_i = 0; v_i < vertex_size; ++v_i) {
		// step 1 - initialization
		// step 1.1 get the relative position of the scene
		float3 d_orig = vertex[v_i];
		int3 pos = make_int3((d_orig - rectgrid.pos_) / rectgrid.interval_);

		// step 1.2 init the step
		int3 step;
		step.x = (d_dir.x >= 0) ? 1 : -1;
		step.y = (d_dir.y >= 0) ? 1 : -1;
		step.z = (d_dir.z >= 0) ? 1 : -1;

		// step 1.3 init the tmax
		float3 t_max, t;
		t.x = cal_t_max(d_dir.x, rectgrid.interval_.x, pos.x, d_orig.x - rectgrid.pos_.x);
		t.y = cal_t_max(d_dir.y, rectgrid.interval_.y, pos.y, d_orig.y - rectgrid.pos_.y);
		t.z = cal_t_max(d_dir.z, rectgrid.interval_.z, pos.z, d_orig.z - rectgrid.pos_.z);

		t_max.x = abs_divide(t.x, d_dir.x);
		t_max.y = abs_divide(t.y, d_dir.y);
		t_max.z = abs_divide(t.z, d_dir.z);

		// step 1.4 initial t_delta
		float3 t_delta;
		t_delta.x = abs_divide(rectgrid.interval_.x, d_dir.x);
		t_delta.y = abs_divide(rectgrid.interval_.y, d_dir.y);
		t_delta.z = abs_divide(rectgrid.interval_.z, d_dir.z);

		// step 2 - intersection
		int3 grid_index = pos;
		int grid_address = global_func::unroll_index(grid_index, rectgrid.grid_num_);

		while (true) {
			// add the grid 
			int block_num = intersect_helio(
				d_orig,
				d_dir,
				grid_address,
				h_heliotat_vertex,
				h_grid_heliostat_match,
				h_grid_heliostat_index,
				relative_helio_label);

#ifdef _DEBUG
			if (block_num) {
				std::cout << "the block number is: " << block_num << std::endl;
			}
#endif
			if (t_max.x < t_max.y)
			{
				if (t_max.x < t_max.z)
				{
					grid_index.x += step.x;
					if (grid_index.x >= rectgrid.grid_num_.x || grid_index.x < 0)
						// Outside grid
						break;
					t_max.x += t_delta.x;
				}
				else
				{
					grid_index.z += step.z;
					if (grid_index.z >= rectgrid.grid_num_.z || grid_index.z < 0)
						// Outside grid
						break;
					t_max.z += t_delta.z;
				}
			}
			else
			{
				if (t_max.y < t_max.z)
				{
					grid_index.y += step.y;
					if (grid_index.y >= rectgrid.grid_num_.y || grid_index.y < 0)
						// Outside grid
						break;
					t_max.y += t_delta.y;
				}
				else
				{
					grid_index.z += step.z;
					if (grid_index.z >= rectgrid.grid_num_.z || grid_index.z < 0)
						// Outside grid
						break;
					t_max.z += t_delta.z;
				}
			}
			grid_address = global_func::unroll_index(grid_index, rectgrid.grid_num_);

//			// add the grid 
//			int block_num = intersect_helio(
//				d_orig,
//				d_dir,
//				grid_address,
//				h_heliotat_vertex,
//				h_grid_heliostat_match,
//				h_grid_heliostat_index,
//				relative_helio_label);
//
//#ifdef _DEBUG
//			std::cout << "the block number is: " << block_num << std::endl;
//#endif
		}
	}
}