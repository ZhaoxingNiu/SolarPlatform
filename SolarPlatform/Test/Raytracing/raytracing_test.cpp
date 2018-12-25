#include "./raytracing_test.cuh"
#include "time.h"

bool test_raytracing()
{
	// 单点与卷积需要修改的位置
	// num_sunshape_lights_per_group
	// helio_pixel_length
	solarenergy::num_sunshape_lights_per_group = 8192;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::scene_filepath = "../SceneData/onepoint/helios_1_4_distance_500.scn";
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;
	
	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();

	vector<int> angle_vec = { 0 }; //  30, 45, 60, 90, 135
	int angel = 0;
	solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));
	// Step 2: Initialize the content in the scene
	solar_scene->InitContent();

	vector<int> helio_vec = { 0, 1, 2, 3 };
	for (int helio_index : helio_vec) {
		// Step 3: 
		//string file_outputname = "../SimulResult/onepoint/one_point_angel" + to_string(angle) + "_distance_" + to_string(i) + "00.txt";
		string file_outputname = "../SimulResult/data/testcpu/sub/sub_"+ std::to_string(helio_index) +".txt";
		raytracing_standard_interface(*solar_scene, helio_index, 0, file_outputname);
	}
	// Finally, destroy solar_scene
	solar_scene->~SolarScene();
	return true;
}



bool test_raytracing_scene1()
{
	// 单点与卷积需要修改的位置
	// num_sunshape_lights_per_group
	// helio_pixel_length
	solarenergy::num_sunshape_lights_per_group = 1024;
	solarenergy::num_sunshape_lights_loop = 100;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::total_time = 0.0f;
	solarenergy::scene_filepath = "../SceneData/paper/helioField_scene1.scn";
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	solarenergy::sun_dir = make_float3(0.0f, -0.867765f, -1.0f);
	
	// Step 2: Initialize the content in the scene
	solar_scene->InitContent();

	//double total_time = 0.0;
	for (int helio_index = 0; helio_index < 40; ++helio_index) {
		// Step 3: 
		string file_outputname = "../SimulResult/paper/scene1/raytracing/"
			+ std::to_string(int(solarenergy::num_sunshape_lights_per_group*solarenergy::num_sunshape_lights_loop))
			+"/equinox_12_#" + std::to_string(helio_index) + ".txt";
		int grid_index = helio_index;

		//double start, stop, durationTime;
		//start = clock();
		raytracing_standard_interface(*solar_scene, helio_index, grid_index, file_outputname);
		//stop = clock();
		//durationTime = ((double)(stop - start)) / CLK_TCK;
		//std::cout << "程序耗时：" << durationTime << " s" << endl;
		//total_time += durationTime;
	}
	std::cout << "程序平均耗时：" << solarenergy::total_time/40 << " s" << endl;
	// Finally, destroy solar_scene
	solar_scene->~SolarScene();
	return true;
}