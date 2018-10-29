#include "./dda_test.h"
#include "../Struct/oblique_parallel.cuh"

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

bool test_dda_rasterization() {
	// h_index, chose the heliostat 
	int h_index = 1;
	int rece_index = 0;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	// set the pixel 
	solarenergy::num_sunshape_lights_per_group = 1024;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::image_plane_pixel_length = 0.05f;
	solarenergy::scene_filepath = "../SceneData/imageplane/face2face_shadow.scn";
	solarenergy::sun_dir = make_float3(0.0f ,0.0f, 1.0f);

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
	RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[0]);
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);
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

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	projection_plane_rect(
		(solar_scene->receivers[0])->d_image_,
		plane.get_deviceData(),
		rectrece,
		&plane,
		M,
		offset);

	sdkStopTimer(&hTimer);

	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("projection cost time: (%f ms)\n", gpuTime);


#ifdef _DEBUG
	std::string receiver_path = "../SimulResult/imageplane/receiver_debug.txt";
	solar_scene->receivers[0]->save_result(receiver_path);
#endif


	delete[] M;

	return true;
}