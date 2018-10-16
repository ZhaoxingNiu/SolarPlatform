#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

class ProjectionPlane {
public:
	ProjectionPlane(
		int rows_, 
		int cols_, 
		float pixel_length_, 
		float row_offset_,
		float col_offset_);

	void getSize(int &rows_, int &cols);
	void setDeviceData(float *d_Data_);
	float* getDeviceData();

	// calculate the projection area
	void projection(const std::vector<float3> &points);

	// calcluate the shadow and blockss
	void shadowBlock(const std::vector<std::vector<float3>> &points);
	
private:
	float *d_Data;
	int rows;
	int cols;
	float pixel_length;
	float row_offset;
	float col_offset;
};