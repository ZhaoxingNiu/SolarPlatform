#include "./gen_kernel.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>


void gen_kernel(
	float true_dis,
	float ori_dis,
	float angel,
	bool flush,
	float step_r,
	float grid_len,
	float distance_threshold,
	float rece_width,
	float rece_height,
	float rece_max_r) {

	// if the kernel is generate, 
	if (flush) {
		int round_angel = round(angel);
		int round_distance = round(true_dis);
		std::string kernel_path = "../SimulResult/data/gen_flux/onepoint_angle_";
		kernel_path += std::to_string(round_angel) + "_distance_"
			+ std::to_string(round_distance) + ".txt";
		std::ifstream fin(kernel_path);
		if (fin) {
			std::cout << kernel_path <<" file  exist " << std::endl;
			return;
		}
	}

	// change the dir 
	std::string command = "cd ../Script/solarEnergy & python interface.py ";
	command += std::to_string(true_dis);
	command += " --ori_dis " + std::to_string(ori_dis);
	command += " --angel " + std::to_string(angel);
	command += " --step_r " + std::to_string(step_r);
	command += " --grid_len " + std::to_string(grid_len);
	command += " --distance " + std::to_string(distance_threshold);
	command += " --rece_width " + std::to_string(rece_width);
	command += " --rece_height " + std::to_string(rece_height);
	command += " --rece_max_r " + std::to_string(rece_max_r);

	std::cout << command << std::endl;
	system(command.c_str());
}