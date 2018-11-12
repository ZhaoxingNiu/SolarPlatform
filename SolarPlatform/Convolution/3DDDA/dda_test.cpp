#include "./dda_test.h"
#include <cmath>


bool test_dda_rasterization() {

	// Step 0: initialization
	// Step 0.1: init the parameters
	// h_index, chose the heliostat 
	int rece_index = 0;
	int helio_index = 0;
	int grid_index = 0;

	float angel = 60.0f;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::image_plane_pixel_length = 0.05f;

	// test receiver
	solarenergy::scene_filepath = "../SceneData/imageplane/face_imageplane.scn";
	solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));

	// test shadow
	// solarenergy::scene_filepath = "../SceneData/imageplane/face_shadow.scn";
	// solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));

	// test receiver
	//solarenergy::scene_filepath = "../SceneData/onepoint/helioField_small.scn";
	//solarenergy::sun_dir = make_float3(0.0f, -0.5f, 0.866025404f);

	// test the sub_heliostat
	// solarenergy::scene_filepath = "../SceneData/onepoint/helios_1_4_distance_500.scn";
	// solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	// Load files and init the scenefiles
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	solar_scene->InitContent();
	solar_scene->receivers[rece_index]->Cclean_image_content();

	//  ÇÐ»» kernel Ñ¡Ïî
	conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, kernelType::T_GAUSSIAN_CONV);
	//conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, kernelType::T_GAUSSIAN_CONV_MATLAB);
	//conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, kernelType::T_LOADED_CONV);

	std::string receiver_path = "../SimulResult/imageplane/receiver_angel_60.txt";
	//std::string receiver_path = "../SimulResult/data/testcpu/sub/conv2_sub_" + std::to_string(helio_index) + ".txt";
	solar_scene->receivers[rece_index]->save_result(receiver_path);

	return true;
}