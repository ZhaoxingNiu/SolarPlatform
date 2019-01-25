#include "./KernelManager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>


KernelManager::KernelManager(std::string base_path, float ori_dis)
	:src_base_path_(base_path),origin_dis_(ori_dis) {
	// set the kernel size
	cols_ = solarenergy::kernel_cols;
	rows_ = solarenergy::kennel_rows;
	width_ = solarenergy::kernel_width;
	height_ = solarenergy::kernel_height;

	step_r_ = solarenergy::kernel_step_r;
	grid_len_ = solarenergy::kernel_grid_len;
	distance_threshold_ = solarenergy::kernel_distance_threshold;
	rece_width_ = solarenergy::kernel_rece_width;
	rece_height_ = solarenergy::kernel_rece_height;
	rece_max_r_ = solarenergy::kernel_rece_max_r;

	MakeSpline();
}


void KernelManager::GetKernel(float dis, float angle, float *kernel) {
    // assert kernel_size is  rows_ cols_
	float air_rate1 = global_func::air_attenuation(origin_dis_);
	float air_rate2 = global_func::air_attenuation(dis);
	float distance_rate = origin_dis_ / dis;
	float attenuation_rate = air_rate2 / air_rate1;
	tk::spline *s = mp_[angle];

	for (int i = 0; i < rows_; ++i) {
		for (int j = 0; j < cols_; ++j) {
			double pos_x = -width_ / 2 + (i + 0.5)*grid_len_;
			double pos_y = -height_ / 2 + (i + 0.5)*grid_len_;
			double pos_r = sqrt(pos_x*pos_x + pos_y*pos_y);
			double true_pos_r = pos_r*distance_rate;
			// TODO: check the code 
			if (true_pos_r < distance_threshold_) {
				kernel[i * cols_ + j] = attenuation_rate*((*s)(true_pos_r))
					* MATH_PI / distance_threshold_ / distance_threshold_;
			}
			else {
				// limit the true_pos's size
				if (true_pos_r > rece_max_r_) {
					true_pos_r = rece_max_r_;
				}
				kernel[i * cols_ + j] = attenuation_rate* ((*s)(true_pos_r, 1))*distance_rate
					/ 2 / MATH_PI / pos_r;
			}
		}
	}
}



void KernelManager::LoadFluxData(std::string path, std::vector<double> &flux_data, double rate) {
	int data_size = flux_data.size();
	std::string str_line;
	std::ifstream flux_file;
	int cnt = 0;
	try {
		flux_file.open(path);
		std::stringstream scene_stream;
		// read file's buffer contents into streams
		scene_stream << flux_file.rdbuf();
		flux_file.close();
		while (getline(scene_stream, str_line)) {
			std::stringstream input(str_line);
			std::string tmp;
			while (getline(input, tmp, ',')) {
				double tmp_val = global_func::stringToFloat(tmp);
				flux_data[cnt++] = tmp_val * rate;
			}
		}
	}
	catch (std::ifstream::failure e)
	{
		std::cerr << "ERROR::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		return;
	}

	if (cnt != data_size) {
		std::cerr << "ERROR::FLUX_DATA_SIZE_NOT_EQUAL" << std::endl;
		return;
	}

}

void KernelManager::GetAccumulate(const std::vector<double> &flux_data, std::vector<double> &energy_dis,
	std::vector<double> &energy_accu) {
	double step_x = grid_len_;
	double step_y = grid_len_;
	double grid_size = grid_len_*grid_len_;

	int flux_rows = int(rece_height_ / grid_len_);
	int flux_cols = int(rece_width_ / grid_len_);

	// init the energy_dis
	int dis_size = int(  rece_max_r_ / step_r_)+1;
	energy_dis.resize(dis_size);
	energy_accu.resize(dis_size);

	energy_dis[0] = 0;
	for (int i = 1; i <= dis_size; ++i) {
		energy_dis[i] = (i - 0.5) * step_r_;
	}

	std::vector<double> energy_static(dis_size);
	for (int i = 0; i < flux_rows; ++i) {
		for (int j = 0; j < flux_cols; ++j) {
			double pos_x = -rece_width_ / 2 + (i + 0.5) * step_x;
			double pos_y = -rece_height_ / 2 + (i + 0.5) * step_y;
			double pos_r = sqrt(pos_x*pos_x + pos_y * pos_y);
			energy_static[int(pos_r / step_r_)] += flux_data[i*flux_cols + j] * grid_size;
		}
	}

	energy_accu[0] = 0.0;
	for (int i = 1; i <= dis_size; ++i) {
		energy_accu[i] = energy_accu[i - 1] + energy_static[i];
	}

}


void KernelManager::MakeSpline() {
	int flux_rows = round(rece_height_/grid_len_);
	int flux_cols = round(rece_width_/grid_len_);
	std::vector<double> flux_data(flux_rows*flux_cols);
	for (int angle = 0; angle < 180; ++angle) {
		// load the kernel
		std::string path = src_base_path_ + "/angle_" + std::to_string(angle) + ".txt";
		float flux_angle_rate = 1.0 / cos(angle / 2 * MATH_PI / 180);
		LoadFluxData(path, flux_data, flux_angle_rate);

		std::vector<double> energy_dis;
		std::vector<double> energy_accu;
		GetAccumulate(flux_data, energy_dis, energy_accu);

		// fit the spline 
		tk::spline *s = new tk::spline();
		s->set_points(energy_dis, energy_accu);
		mp_[angle] = s;
	}
}