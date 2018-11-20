#include "./dda_test.h"
#include <cmath>


bool test_dda_rasterization() {

	// Step 0: initialization
	// Step 0.1: init the parameters
	// h_index, chose the heliostat
	int rece_index = 0;
	int helio_index = 3;
	int grid_index = 0;
	// set the normal
	bool set_image_normal = false;

	float angel = 30.0f;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::image_plane_pixel_length = 0.05f;

	// test receiver
	// solarenergy::scene_filepath = "../SceneData/imageplane/face_imageplane.scn";
	// solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));

	// test shadow
	// solarenergy::scene_filepath = "../SceneData/imageplane/face_shadow.scn";
	// solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));

	// test receiver
	//solarenergy::scene_filepath = "../SceneData/onepoint/helioField_small.scn";
	//solarenergy::sun_dir = make_float3(0.0f, -0.5f, 0.866025404f);

	// test the sub_heliostat
	solarenergy::scene_filepath = "../SceneData/onepoint/helios_1_4_distance_500.scn";
	solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	// Load files and init the scenefiles
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	solar_scene->InitContent();
	solar_scene->receivers[rece_index]->Cclean_image_content(); 

	//  ÇÐ»» kernel Ñ¡Ïî
	//conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f, 0.0f, 0.0f),kernelType::T_GAUSSIAN_CONV, 1.2f);
	//conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f,0.0f,0.0f), kernelType::T_GAUSSIAN_CONV_MATLAB);
	//conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f,0.0f,0.0f), kernelType::T_LOADED_CONV);

	float3 image_normal = normalize( make_float3(0.0f,20.0f,500.0f)- solar_scene->focus_center_);
	if (set_image_normal) {  // the sub_heliostat using the same_normal
		conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, image_normal, kernelType::T_LOADED_CONV, 1.2f);
	}
	else {    
		conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f, 0.0f, 0.0f), kernelType::T_LOADED_CONV, 1.2f);
	}

	//std::string receiver_path = "../SimulResult/imageplane/receiver_angel_60.txt";
	std::string receiver_path = "../SimulResult/data/testcpu/sub/conv1_sub_" + std::to_string(helio_index) + ".txt";
	solar_scene->receivers[rece_index]->save_result_conv(receiver_path);

	return true;
}