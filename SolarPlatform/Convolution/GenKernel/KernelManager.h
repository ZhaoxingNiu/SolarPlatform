#pragma once
#ifndef KERNEL_MANAGER_H
#define KERNEL_MANAGER_H

#include <unordered_map>
#include <vector>

#include "../../Common/common_var.h"
#include "../../Common/global_constant.h"
#include "../../Common/global_function.cuh"
#include "./spline.h"

class KernelManager {
public:
	//need set the default valye  dni,csr,disturb_std,reflected_rate
	KernelManager(std::string base_path, float ori_dis); 
	void GetKernel(float dis, float angle, float *kernel);

private:
	void MakeSpline();
	void LoadFluxData(std::string path, std::vector<double> &flux_data, double rate);
	void GetAccumulate(const std::vector<double> &flux_data, std::vector<double> &energy_dis, 
		std::vector<double> &energy_accu);

	std::string src_base_path_;
	float origin_dis_;
	int cols_;
	int rows_;
	float width_;
	float height_;

	float step_r_;
	float grid_len_;
	float distance_threshold_;
	float rece_width_;
	float rece_height_;
	float rece_max_r_;

	std::unordered_map<double, tk::spline *> mp_;
	
};



#endif // KERNEL_MANAGER_H