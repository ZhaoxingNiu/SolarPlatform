#include "./raytracing_test.cuh"
#include "../../SceneProcess/scene_file_proc.h"
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

bool test_raytracing_onepoint()
{
	// 单点与卷积需要修改的位置
	// num_sunshape_lights_per_group
	// helio_pixel_length
	solarenergy::num_sunshape_lights_per_group = 1024000;
	solarenergy::num_sunshape_lights_loop = 100;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 1.0f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::scene_filepath = "../SceneData/onepoint/one_point_odd_new.scn";
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	for (int angel = 0; angel < 180; ++angel) {
		solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));
		// Step 2: Initialize the content in the scene
		solar_scene->InitContent();
		int helio_index = 4;
		// Step 3: 
		string file_outputname = "../SimulResult/data/gen_flux_ori/500/angle_" + to_string(angel)  + ".txt";
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
	/******修改*****/
	//solarenergy::scene_filepath = "../SceneData/paper/helioField_scene1.scn";
	solarenergy::scene_filepath = "../SceneData/paper/helioField_scene_shadow.scn";
	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;

	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	//solarenergy::sun_dir = make_float3(0.0f, -0.867765f, -1.0f);
	solarenergy::sun_dir = make_float3(0.0f, 0.0f, 1.0f);

	// Step 2: Initialize the content in the scene
	solar_scene->InitContent();

	//double total_time = 0.0;
	// *********修改******* /
	for (int helio_index = 0; helio_index < 1; ++helio_index) {
		// Step 3: 
		// *********修改******* /
		string file_outputname = "../SimulResult/paper/scene_shadow/raytracing/"
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


bool test_raytracing_scene_ps10()
{
	// 单点与卷积需要修改的位置
	// num_sunshape_lights_per_group
	// helio_pixel_length
	solarenergy::num_sunshape_lights_per_group = 2048;
	solarenergy::num_sunshape_lights_loop = 1;
	int ray_num = int(solarenergy::num_sunshape_lights_per_group*solarenergy::num_sunshape_lights_loop);
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;

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

	// 计时，并且修改统计方式
	solarenergy::total_time = 0.0f;
	solarenergy::total_times = 624;
	// *********修改******* /
	for (int helio_index = 0; helio_index < solarenergy::total_times * 28; ++helio_index) {
		// Step 3: 
		// *********修改******* /
		string res_path = "../SimulResult/paper/scene_ps10_flat/raytracing/"
			+ std::to_string(ray_num) + "/equinox_12_#"
			+ std::to_string(helio_index/28) + "_" + std::to_string(helio_index % 28)  + ".txt";
		int grid_index = 0;
		raytracing_standard_interface(*solar_scene, helio_index, grid_index, res_path);
	}

	std::cout << "程序平均耗时：" << solarenergy::total_time/ solarenergy::total_times << " s" << endl;
	// Finally, destroy solar_scene
	solar_scene->~SolarScene();
	return true;
}