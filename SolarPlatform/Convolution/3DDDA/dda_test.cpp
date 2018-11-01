#include "./dda_test.h"
#include "../Struct/oblique_parallel.cuh"
#include "../Struct/convKernel.h"
#include "../Cufft/convolutionFFT2D_interface.h"

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cmath>


bool test_dda_rasterization() {
	// h_index, chose the heliostat 
	int helio_index = 14;
	int rece_index = 0;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	// set the pixel 
	solarenergy::num_sunshape_lights_per_group = 10240;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::image_plane_pixel_length = 0.05f;

	// test shadow
	//solarenergy::scene_filepath = "../SceneData/imageplane/face_shadow.scn";
	//solarenergy::sun_dir = make_float3(sin(angel*MATH_PI / 180), 0.0f, cos(angel*MATH_PI / 180));

	// test receiver
	solarenergy::scene_filepath = "../SceneData/onepoint/helioField_small.scn";
	solarenergy::sun_dir = make_float3(0.0f, -0.5f, 0.866025404f);

	float angel = 0.0f;
	int round_angel = round(angel);

	std::cout << "filepath: " << solarenergy::scene_filepath << std::endl;
	// Step 1: Load files
	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();

	// Step 2: Initialize the content and set the image plane
	solar_scene->InitContent();

	int2 plane_size;
	plane_size.x = 200;
	plane_size.y = 200;

	ProjectionPlane plane(
		plane_size.x, plane_size.y,
		solarenergy::image_plane_pixel_length);
	
	// receiver 0 
	RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[helio_index]);
	
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);
	solar_scene->receivers[rece_index]->Cclean_image_content();
	// get normal
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	float3 out_dir = reflect(in_dir, recthelio->normal_);   // reflect light
	out_dir = normalize(out_dir);
	plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, -out_dir);

	// Step 3: rasterization
	dda_interface(
		*(solar_scene->sunray_),
		plane,
		*recthelio,
		*(solar_scene->grid0s[0]),
		solar_scene->heliostats
	);

#ifdef _DEBUG
	std::string image_path = "../SimulResult/imageplane/image_debug.txt";
	plane.save_data_text(image_path);
#endif
	
	// Step 4: projection the image plane to the heliostat
	// Step 4.1: get the projection matrix
	float *M = new float[9];
	float3 offset;
	oblique_proj_matrix(
		out_dir,
		plane.normal,
		out_dir,
		M,
		offset
	);

	// load the kernel

	std::string kernel_path = "../SimulResult/data/gen_flux/onepoint_angle_"+
		std::to_string(round_angel) +"_distance_500.txt";
	LoadedConvKernel kernel(201, 201, kernel_path);
	kernel.genKernel();

	fastConvolutionDevice(
		plane.get_deviceData(),
		kernel.d_data,
		plane.rows,
		plane.cols,
		kernel.dataH,
		kernel.dataW
	);
	

#ifdef _DEBUG
	std::string image_path2 = "../SimulResult/imageplane/image_debug2.txt";
	plane.save_data_text(image_path2);
#endif
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	projection_plane_rect(
		(solar_scene->receivers[rece_index])->d_image_,
		plane.get_deviceData(),
		rectrece,
		&plane,
		M,
		offset);

	sdkStopTimer(&hTimer);

	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("projection cost time: (%f ms)\n", gpuTime);


	std::string receiver_path = "../SimulResult/imageplane/receiver_debug_"+ std::to_string(helio_index) +".txt";
	solar_scene->receivers[rece_index]->save_result(receiver_path);
	delete[] M;

	return true;
}