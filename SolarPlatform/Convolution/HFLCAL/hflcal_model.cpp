#include "./hflcal_model.h"

// chose the best
void hflcal_model(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float ideal_peak,
	std::string res_path
) {
	// initialization,set the value
	float ideal_sigma_2 = -1.0;

	float sigma_low = 0.1f;
	float sigma_high = 10.0f;
	float sigma_2;
	// choose the best sigma_2
	while (abs(sigma_high - sigma_low) > 0.01f) {
		// init the clean
		sigma_2 = (sigma_high + sigma_low) / 2.0f;
		solar_scene->receivers[rece_index]->Cclean_image_content();
		conv_method_kernel_HFLCAL(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f, 0.0f, 0.0f),
			sigma_2);
		float max_val = solar_scene->receivers[rece_index]->peek_value();
		if (abs(max_val - ideal_peak) < 0.1f) {
			// break
			sigma_high = sigma_high = sigma_2;
		}
		else if (max_val > ideal_peak) {
			sigma_low = sigma_2;
		}
		else {
			sigma_high = sigma_2;
		}
	}
	ideal_sigma_2 = (sigma_high + sigma_low) / 2.0f;

	// run the convolution model
	solar_scene->receivers[rece_index]->Cclean_image_content();
	conv_method_kernel_HFLCAL(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f, 0.0f, 0.0f),
		 ideal_sigma_2);

	// save the result
	solar_scene->receivers[rece_index]->save_result_conv(res_path);
}

bool test_hflcal_model() {
	int rece_index = 0;
	int helio_index = 0;
	int grid_index = 0;
	float ideal_peak = 180.0f;  // 60: 219   135: 150  
	float total_energy = 880.0f;
	float angel = 100.0f;
	int round_angel = angel;
	std::string res_path = "../SimulResult/imageplane/receiver_angel_"
		+ to_string(round_angel) + ".txt";

	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::image_plane_pixel_length = 0.05f;

	// Load files and init the scenefiles
	solarenergy::scene_filepath = "../SceneData/imageplane/face_imageplane.scn";
	solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	solar_scene->InitContent();
	solar_scene->receivers[rece_index]->Cclean_image_content();

	// unizar model
	hflcal_model(solar_scene, rece_index, helio_index, grid_index, ideal_peak, res_path);

	return true;
}


bool test_hflcal_model_scene1() {
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
	solarenergy::sun_dir = make_float3(0.0f, -0.867765f, -1.0f);
	solar_scene = SolarScene::GetInstance();
	solar_scene->InitContent();

	int rece_index = 0;
	for (int helio_index = 0; helio_index < 40; ++helio_index) {
		// clean the receiver
		solar_scene->receivers[rece_index]->Cclean_image_content();
		string raytracing_path = "../SimulResult/paper/scene1/raytracing/102400/equinox_12_#" + std::to_string(helio_index) + ".txt";
		string res_path = "../SimulResult/paper/scene1/hflcal/equinox_12_#" + std::to_string(helio_index) + ".txt";
		float ideal_peak = get_file_peak(raytracing_path);
		int grid_index = helio_index;
		// hfalcal model
		hflcal_model(solar_scene, rece_index, helio_index, grid_index, ideal_peak, res_path);
	}

	std::cout << "程序平均耗时：" << solarenergy::total_time / 40 << " ms" << endl;
	return true;
}