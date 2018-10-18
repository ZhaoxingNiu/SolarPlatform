#include "./projectionPlane.h"
#include "../../Common/global_constant.h"
#include "../Rasterization/rasterization_common.h"
#include "../../Common/vector_arithmetic.cuh"
#include <iostream>

ProjectionPlane::ProjectionPlane(
	int rows_,
	int cols_,
	float pixel_length_,
	float row_offset_,
	float col_offset_):
	d_Data(nullptr), rows(rows_), cols(cols_),
	pixel_length(pixel_length_), row_offset(row_offset_), col_offset(col_offset_) {
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

void ProjectionPlane::set_deviceData(float *d_Data_) {
	d_Data = d_Data_;
}

float* ProjectionPlane::get_deviceData() {
	return d_Data;
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

