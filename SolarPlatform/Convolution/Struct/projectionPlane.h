#ifndef PROJECTION_PLANE_H
#define PROJECTION_PLANE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

class ProjectionPlane {
public:
	ProjectionPlane(
		int rows_, 
		int cols_, 
		float pixel_length_);

	// define the image plane relative position
	void set_pos(float3 pos_, float3 normal_);

	void get_size(int &rows_, int &cols);
	void set_deviceData(float *d_Data_);
	float* get_deviceData();

	// ray intersect
	bool ray_intersect(const float3 ori, const float3 dir, float3 &p) const;

	// ray intersect pos, the relative position
	bool ray_intersect_pos2(const float3 ori, const float3 dir, float3 &p) const;

	// calculate the projection area
	void projection(const std::vector<float3> &points);

	// calcluate the shadow and blockss
	void shadow_block(const std::vector<std::vector<float3>> &points);
	

private:
	float *d_Data;
	int rows;
	int cols;
	float pixel_length;
	float row_offset;
	float col_offset;
	// image plane geometric information
	float3 pos;
	float3 normal;
	float3 v_axis;   // Parallel to the ground
	float3 u_axis;   // perpendicular to the v_axis
};

#endif // !PROJECTION_PLANE_H