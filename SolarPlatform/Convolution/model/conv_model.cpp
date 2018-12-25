#include "./conv_model.h"
#include <cmath>


bool test_conv_model_scene1() {

	// Step 0: initialization
	// Step 0.1: init the parameters
	// h_index, chose the heliostat
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::total_time = 0.0f;
	solarenergy::scene_filepath = "../SceneData/paper/helioField_scene1.scn";
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	// step 2:Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	solarenergy::sun_dir = make_float3(0.0f, -0.867765f, -1.0f);
	solar_scene->InitContent();

	int rece_index = 0;
	for (int helio_index = 0; helio_index < 40; ++helio_index) {
		// clean the receiver
		solar_scene->receivers[rece_index]->Cclean_image_content();

		int grid_index = helio_index;
		// run the model 
		conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f, 0.0f, 0.0f), kernelType::T_LOADED_CONV, 0.0f);

		string file_outputname = "../SimulResult/paper/scene1/model/equinox_12_#" + std::to_string(helio_index) + ".txt";
		solar_scene->receivers[rece_index]->save_result_conv(file_outputname);

	}

	std::cout << "程序平均耗时：" << solarenergy::total_time / 40 << " ms" << endl;
	return true;
}