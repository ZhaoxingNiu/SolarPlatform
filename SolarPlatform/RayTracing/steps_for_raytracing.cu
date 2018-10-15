#include "./steps_for_raytracing.h"

// Step 1: Generate local micro-heliostats' centers
__global__ void map_microhelio_centers(float3 *d_microhelio_centers, float3 helio_size,
	const int2 row_col, const int2 sub_row_col,
	const float2 gap,
	const float pixel_length, const float2 subhelio_rowlength_collength, const size_t size)
{
	unsigned long long int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	int row = myId / (row_col.y*sub_row_col.y);
	int col = myId % (row_col.y*sub_row_col.y);

	int block_row = row / sub_row_col.x;
	int block_col = col / sub_row_col.y;

	d_microhelio_centers[myId].x = col*pixel_length + block_col*gap.x + pixel_length / 2 - helio_size.x / 2;
	d_microhelio_centers[myId].y = helio_size.y / 2;
	d_microhelio_centers[myId].z = row*pixel_length + block_row*gap.y + pixel_length / 2 - helio_size.z / 2;
}

// Step 2: Generate micro-heliostats' normals
__global__ void map_microhelio_normals(float3 *d_microhelio_normals, const float3 *d_microhelio_centers,
	float3 normal,
	const size_t size)
{
	unsigned long long int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	d_microhelio_normals[myId] = normal;
}

// Step 3: Transform local micro-helio center to world postion
__global__ void map_microhelio_center2world(float3 *d_microhelio_world_centers, float3 *d_microhelio_local_centers,
	const float3 normal, const float3 world_pos,
	const size_t size)
{
	unsigned long long int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	float3 local = d_microhelio_local_centers[myId];
	local = global_func::local2world(local, normal);		// Then Rotate
	local = global_func::transform(local, world_pos);		// Translation to the world system
	d_microhelio_world_centers[myId] = local;
}

bool set_microhelio_centers(const RectangleHelio &recthelio, float3 *&d_microhelio_centers, float3 *&d_microhelio_normals, size_t &size)
{
	int2 row_col = recthelio.row_col_;
	float3 helio_size = recthelio.size_;
	float2 gap = recthelio.gap_;
	float pixel_length = recthelio.pixel_length_;

	//int2 row_col = make_int2(2, 4);
	//float3 helio_size = make_float3(10.0f, 0.1f, 8.0f);
	//float2 gap = make_float2(1.0f, 0.8f);
	//float pixel_length = 0.05f;

	float2 subhelio_rowlength_collength;
	subhelio_rowlength_collength.x = (helio_size.z - gap.y*(row_col.x - 1)) / float(row_col.x);
	subhelio_rowlength_collength.y = (helio_size.x - gap.x*(row_col.y - 1)) / float(row_col.y);

	int2 sub_row_col;
	sub_row_col.x = subhelio_rowlength_collength.x / pixel_length;
	sub_row_col.y = subhelio_rowlength_collength.y / pixel_length;

	size = sub_row_col.x*sub_row_col.y*row_col.x*row_col.y;

	int nThreads;
	dim3 nBlocks;
	global_func::setThreadsBlocks(nBlocks, nThreads, size);

	// 1. local center position
	if (d_microhelio_centers == nullptr)
		cudaMalloc((void **)&d_microhelio_centers, sizeof(float3)*size);
	map_microhelio_centers << <nBlocks, nThreads >> >
		(d_microhelio_centers, helio_size, row_col, sub_row_col, gap, pixel_length, subhelio_rowlength_collength, size);

	// 2. normal
	if (d_microhelio_normals == nullptr)
		cudaMalloc((void **)&d_microhelio_normals, sizeof(float3)*size);
	map_microhelio_normals <<<nBlocks, nThreads >>>(d_microhelio_normals, d_microhelio_centers, recthelio.normal_, size);

	// 3. world center position
	map_microhelio_center2world <<<nBlocks, nThreads >>>(d_microhelio_centers, d_microhelio_centers, recthelio.normal_, recthelio.pos_, size);

	return true;
}


// const float3 *d_helio_vertexs
bool set_helios_vertexes(vector<Heliostat *> heliostats, const int start_pos, const int end_pos,
	float3 *&d_helio_vertexs)
{
	int size = end_pos-start_pos;
	float3 *h_helio_vertexes = new float3[size * 3];

	for (int i = start_pos; i < end_pos; ++i)
	{
		int j = i - start_pos;
		heliostats[i]->Cget_vertex(h_helio_vertexes[3 * j], h_helio_vertexes[3 * j + 1], h_helio_vertexes[3 * j + 2]);
	}
		
	
	global_func::cpu2gpu(d_helio_vertexs, h_helio_vertexes,  3 * size);

	delete[] h_helio_vertexes;
	return true;
}

// int *d_microhelio_groups
bool set_microhelio_groups(int *&d_microhelio_groups, const int num_group, const size_t &size)
{
	if (d_microhelio_groups == nullptr)
		checkCudaErrors(cudaMalloc((void **)&d_microhelio_groups, sizeof(int)*size));

	RandomGenerator::gpu_Uniform(d_microhelio_groups, 0, num_group, size);
	return true;
}