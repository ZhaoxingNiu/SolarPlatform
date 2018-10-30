#include "./projectionPlane.h"

#include "../../Common/utils.h"
#include "../../Common/global_constant.h"
#include "../../Common/vector_arithmetic.cuh"
#include "../../Common/global_function.cuh"
#include "../Rasterization/rasterization_common.h"

#include <iostream>
#include <fstream>
#include <string>

ProjectionPlane::ProjectionPlane(
	int rows_,
	int cols_,
	float pixel_length_):
	d_Data(nullptr), rows(rows_), cols(cols_),
	pixel_length(pixel_length_) {
	
	row_offset = -(rows - 1) * pixel_length / 2;
	col_offset = -(cols - 1) * pixel_length / 2;

	checkCudaErrors(cudaMalloc((void **)&d_Data, rows * cols * sizeof(float)));
	checkCudaErrors(cudaMemset(d_Data, 0.0, rows * cols * sizeof(float)));

}

void ProjectionPlane::set_pos(float3 pos_, float3 normal_) {
	pos = pos_;
	normal = normalize(normal_);
	// the normal 
	if (abs(normal.x) < Epsilon && abs(normal.z) < Epsilon) {
		v_axis = make_float3(1.0f, 0.0f, 0.0f);
		u_axis = make_float3(0.0f, 0.0f, 1.0f);
		std::cerr << "the normal should not be the y axis" << std::endl;
	}
	else {
		v_axis = cross(make_float3(0.0f, 1.0f, 0.0f), normal);
		v_axis = normalize(v_axis);
		u_axis = cross(normal, v_axis);
		u_axis = normalize(u_axis);
	}
}

void ProjectionPlane::get_size(int &rows_,int &cols_) {
	rows_ = rows;
	cols_ = cols;
}


float* ProjectionPlane::get_deviceData() {
	return d_Data;
}

// get the intersection point
bool ProjectionPlane::ray_intersect(const float3 ori, const float3 dir, float3 &p) const{
	float t = (dot(normal,pos) - dot(normal,ori)) / dot(normal,dir);
	if (t < 0) {
		return false;
	}
	p = ori + t * dir;
	return true;
}

// get the intersection point
bool ProjectionPlane::ray_intersect_pos2(const float3 ori, const float3 dir, float3 &p) const {
	float t = (dot(normal, pos) - dot(normal, ori)) / dot(normal, dir);
	if (t < 0) {
		return false;
	}
	float3 relative_pos = ori + t * dir - pos;
	p.x = dot(relative_pos, v_axis);
	p.y = dot(relative_pos, u_axis);
	p.z = 0;
	return true;
}

// 阴影和遮挡暂时只考虑矩形的定日镜，其他形状不考虑
void ProjectionPlane::projection(const std::vector<float3> &points) {
	// project 
	triangle_rasterization(d_Data, rows, cols, pixel_length, row_offset, col_offset,
		points[0], points[1], points[2], points[3], 1.0f);
}


void ProjectionPlane::shadow_block(const std::vector<std::vector<float3>> &points) {
	// shadow and block
	int shadow_num = points.size();
	for (int i = 0; i < shadow_num; ++i) {
		triangle_rasterization(d_Data, rows, cols, pixel_length, row_offset, col_offset,
			points[i][0], points[i][1], points[i][2], points[i][3], 0.0f);
	}
}


void ProjectionPlane::save_data_text(const std::string out_path) {

	std::ofstream out(out_path.c_str());
	// show data 
	float *h_Data = (float *)malloc(rows * cols * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_Data, d_Data, rows * cols * sizeof(float),
		cudaMemcpyDeviceToHost));
	for (int i = 0; i < rows; i += 1) {
		for (int j = 0; j < cols; j += 1) {
			if (j) {
				out << "," << h_Data[i * cols + j];
			}
			else {
				out << h_Data[i * cols + j];
			}
		}
		out << std::endl;
	}

	// free the data
	free(h_Data);
}

ProjectionPlane::~ProjectionPlane() {
	if (d_Data)
	{
		checkCudaErrors(cudaFree(d_Data));
		d_Data = nullptr;
	}
}

