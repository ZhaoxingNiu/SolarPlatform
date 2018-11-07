#include "./raytracing_test.cuh"


bool test_raytracing()
{
	// 单点与卷积需要修改的位置
	// num_sunshape_lights_per_group
	// helio_pixel_length
	solarenergy::num_sunshape_lights_per_group = 4096;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::scene_filepath = "../SceneData/onepoint/helios_1_4_distance_500.scn";

	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;
	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();

	vector<int> angle_vec = { 30 }; //  30, 45, 60, 90, 135
	vector<int> helio_vec = { 0, 1, 2, 3 };
	for (int helio_index : helio_vec) {
		int angel = 30;
		solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));
		// Step 2: Initialize the content in the scene
		solar_scene->InitContent();
		// Step 3: 
		//string file_outputname = "../SimulResult/onepoint/one_point_angel" + to_string(angle) + "_distance_" + to_string(i) + "00.txt";
		string file_outputname = "../SimulResult/data/testcpu/sub/sub_"+ std::to_string(helio_index) +".txt";
		raytracing_standard_interface(*solar_scene, helio_index, 0, file_outputname);
	}
	// Finally, destroy solar_scene
	solar_scene->~SolarScene();
	return true;
}