#include "./gen_kernel.h"
#include <string>
#include <iostream>

void gen_kernel(
	float distance,
	float angel,
	float step_r,
	float grid_len,
	float distance_threshold,
	float rece_width,
	float rece_height,
	float rece_max_r) {

	// change the dir 
	std::string command = "cd ../Script/solarEnergy & python interface.py ";
	command += std::to_string(distance);
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