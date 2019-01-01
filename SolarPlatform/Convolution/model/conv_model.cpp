#include "./conv_model.h"
#include <cmath>
#include "../../SceneProcess/scene_file_proc.h"


bool test_conv_model_scene1() {

	// Step 0: initialization
	// Step 0.1: init the parameters
	// h_index, chose the heliostat
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::total_time = 0.0f;
	/******修改*****/
	//solarenergy::scene_filepath = "../SceneData/paper/helioField_scene1.scn";
	//solarenergy::scene_filepath = "../SceneData/paper/helioField_scene_shadow.scn";

	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	// step 2:Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	solarenergy::sun_dir = make_float3(0.0f, -0.867765f, -1.0f);
	//solarenergy::sun_dir = make_float3(0.0f, 0.0f, 1.0f);
	solar_scene->InitContent();

	int rece_index = 0;
	double first_times = 0;
	for (int helio_index = 0; helio_index < 40; ++helio_index) {
		std::cout << "*********************" << endl;
		// clean the receiver
		solar_scene->receivers[rece_index]->Cclean_image_content();

		int grid_index = helio_index;
		// run the model 
		conv_method_kernel(solar_scene, rece_index, helio_index, grid_index, make_float3(0.0f, 0.0f, 0.0f), kernelType::T_LOADED_CONV, 0.0f);

		// *********修改******* /
		string file_outputname = "../SimulResult/paper/scene_11/model/equinox_12_#" + std::to_string(helio_index) + ".txt";
		solar_scene->receivers[rece_index]->save_result_conv(file_outputname);
		if (helio_index == 0) {
			first_times = solarenergy::total_time;
		}
	}

	std::cout << "程序平均耗时：" << (solarenergy::total_time - first_times) / 39 << " ms" << endl;
	return true;
}

// 这个只是为论文的临时修改版
// 28个平面分别计算，然后进行累加
bool test_conv_model_scene_ps10_tmp() {
	// num_sunshape_lights_per_group
	// helio_pixel_length
	solarenergy::num_sunshape_lights_per_group = 1024;
	solarenergy::num_sunshape_lights_loop = 1;
	int ray_num = int(solarenergy::num_sunshape_lights_per_group*solarenergy::num_sunshape_lights_loop);
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::total_time = 0.0f;
	/******修改*****/
	solarenergy::scene_filepath = "../SceneData/paper/ps10/ps10_flat_rece_split_1.scn";
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	//load the norm
	std::string normal_filepath = "../SceneData/paper/ps10/ps10_flat_rece_split_1_norm.scn";
	std::cout << "filepath: " << normal_filepath << std::endl;
	std::vector<float3> norm_vec;
	SceneFileProc::SceneNormalRead(normal_filepath, norm_vec);


	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	// set the normal
	solarenergy::sun_dir = make_float3(0.0f, -0.79785f, -1.0f);
	solarenergy::sun_dir = normalize(solarenergy::sun_dir);

	// Step 2: Initialize the content in the scene
	// 有两种方式，后续调整，也可以直接进行调整
	solar_scene->InitContent();
	solar_scene->ResetHelioNorm(norm_vec);

	int rece_index = 0;
	solarenergy::total_time = 0.0f;
	solarenergy::total_times = 1;
	double first_times = 0;
	for (int helio_index = 0; helio_index < solarenergy::total_times*28; ++helio_index) {
		// clean the receiver
		solar_scene->receivers[rece_index]->Cclean_image_content();

		int grid_index = 0;
		// run the model 
		conv_method_kernel(solar_scene, rece_index, helio_index, 0, make_float3(0.0f, 0.0f, 0.0f), kernelType::T_LOADED_CONV, 0.0f);

		// *********修改******* /
		string file_outputname = "../SimulResult/paper/scene_ps10_flat/model_sub_tmp/equinox_12_#"
			+ std::to_string(helio_index / 28) + "_" + std::to_string(helio_index % 28) + ".txt";
		solar_scene->receivers[rece_index]->save_result_conv(file_outputname);
		if (helio_index == 0) {
			first_times = solarenergy::total_time;
		}
	}

	std::cout << "程序平均耗时：" << (solarenergy::total_time - first_times) / solarenergy::total_times << " ms" << endl;
	return true;
}

// 论文修改版
// 28个平面使用平均法向平面进行处理，然后得到总体结果
bool test_conv_model_scene_ps10() {
	// num_sunshape_lights_per_group
	// helio_pixel_length
	solarenergy::num_sunshape_lights_per_group = 1024;
	solarenergy::num_sunshape_lights_loop = 1;
	int ray_num = int(solarenergy::num_sunshape_lights_per_group*solarenergy::num_sunshape_lights_loop);
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::total_time = 0.0f;
	/******修改*****/
	solarenergy::scene_filepath = "../SceneData/paper/ps10/ps10_flat_rece_split_1.scn";
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	//load the norm
	std::string normal_filepath = "../SceneData/paper/ps10/ps10_flat_rece_split_1_norm.scn";
	std::cout << "filepath: " << normal_filepath << std::endl;
	std::vector<float3> norm_vec;
	SceneFileProc::SceneNormalRead(normal_filepath, norm_vec);


	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	// set the normal
	solarenergy::sun_dir = make_float3(0.0f, -0.79785f, -1.0f);
	solarenergy::sun_dir = normalize(solarenergy::sun_dir);

	// Step 2: Initialize the content in the scene
	// 有两种方式，后续调整，也可以直接进行调整
	solar_scene->InitContent();
	solar_scene->ResetHelioNorm(norm_vec);

	int rece_index = 0;
	solarenergy::total_time = 0.0f;
	int helio_index_range = 5;
	double first_times = 0;
	for (int helio_index = 4; helio_index < helio_index_range; ++helio_index) {
		// clean the receiver
		solar_scene->receivers[rece_index]->Cclean_image_content();

		int grid_index = 0;
		// run the model 
		conv_method_kernel_focus(solar_scene, rece_index, helio_index, 28, 0, kernelType::T_LOADED_CONV, 0.0f);

		// *********修改******* /
		string file_outputname = "../SimulResult/paper/scene_ps10_flat/model_sub_tmp2/equinox_12_#"
			+ std::to_string(helio_index ) + ".txt";
		solar_scene->receivers[rece_index]->save_result_conv(file_outputname);
		if (helio_index == 0) {
			first_times = solarenergy::total_time;
		}
	}

	std::cout << "程序平均耗时：" << (solarenergy::total_time - first_times) / (helio_index_range -1+0.0001) << " ms" << endl;
	return true;
}