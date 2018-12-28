#ifndef FOCUSHELIOSSPLITS_H
#define FOCUSHELIOSSPLITS_H

#include "../../SceneProcess/solar_scene.h"
#include "../../SceneProcess/scene_file_proc.h"
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#define FOCUS_LENGTH_RATE 1.0

class FocusHeliosSplit {
public:
	void init(SolarScene* solar_scene);
	void split();
	void saveFile(std::string file_out_pos, std::string file_out_norm);

	SolarScene* solar_scene_;
	int helio_num;
	int sub_num;

	int2 row_col;
	float2 gap_length;
	float3 helio_size;
	float3 helio_sub_size;

	std::vector<float3> sub_pos;
	std::vector<float3> sub_size;
	std::vector<float3> sub_normal;
	std::vector<float3> sub_vertex;
	
	std::vector<float> focus_length;

private:
	void local_coor();
	void set_surface();
	void move_to_surface();
	void rotate();
	void transform();
	
};


bool test_focus_helios_split();


#endif