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
	solarenergy::scene_filepath = "../SceneData/imageplane/face_shadow.scn";

	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;
	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();

	vector<int> angle_vec = { 0 }; //  30, 45, 60, 90, 135
	for (int angle : angle_vec) {
		solarenergy::sun_dir = make_float3(sin(angle*MATH_PI / 180), 0.0f, cos(angle*MATH_PI / 180));
		// Step 2: Initialize the content in the scene
		solar_scene->InitContent();
		// Step 3: 
		//string file_outputname = "../SimulResult/onepoint/one_point_angel" + to_string(angle) + "_distance_" + to_string(i) + "00.txt";
		string file_outputname = "../SimulResult/raytracing/shadow_test.txt";
		raytracing_standard_interface(*solar_scene, 0, 0, file_outputname);
	}
	// Finally, destroy solar_scene
	solar_scene->~SolarScene();
	return true;
}