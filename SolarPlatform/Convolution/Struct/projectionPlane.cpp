#include "./projectionPlane.h"
#include "../Rasterization/rasterization_common.h"


ProjectionPlane::ProjectionPlane(
	int rows_,
	int cols_,
	float pixel_length_,
	float row_offset_,
	float col_offset_):
	d_Data(nullptr), rows(rows_), cols(cols_),
	pixel_length(pixel_length_), row_offset(row_offset_), col_offset(col_offset_) {
}

void ProjectionPlane::getSize(int &rows_,int &cols_) {
	rows_ = rows;
	cols_ = cols;
}

void ProjectionPlane::setDeviceData(float *d_Data_) {
	d_Data = d_Data_;
}

float* ProjectionPlane::getDeviceData() {
	return d_Data;
}

// 阴影和遮挡暂时只考虑矩形的定日镜，其他形状不考虑
void ProjectionPlane::projection(const std::vector<float3> &points) {
	// project 
	triangle_rasterization(d_Data, rows, cols, pixel_length, row_offset, col_offset,
		points[0], points[1], points[2], points[3], 1.0f);
}


void ProjectionPlane::shadowBlock(const std::vector<std::vector<float3>> &points) {
	// shadow and block
	int shadow_num = points.size();
	for (int i = 0; i < shadow_num; ++i) {
		triangle_rasterization(d_Data, rows, cols, pixel_length, row_offset, col_offset,
			points[i][0], points[i][1], points[i][2], points[i][3], 0.0f);
	}
}

