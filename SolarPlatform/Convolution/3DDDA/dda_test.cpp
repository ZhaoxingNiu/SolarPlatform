#include "./dda_test.h"

void test_dda_rasterization() {
	// h_index, chose the heliostat 
	int h_index = 1;

	// set the pixel 
	solarenergy::num_sunshape_lights_per_group = 1024;
	solarenergy::csr = 0.1f;
	solarenergy::disturb_std = 0.001f;
	solarenergy::helio_pixel_length = 0.01f;
	solarenergy::receiver_pixel_length = 0.05f;
	solarenergy::image_plane_pixel_length = 0.05f;
	solarenergy::scene_filepath = "../SceneData/conv/one_point_odd.scn";
	solarenergy::sun_dir = make_float3(1.0f ,0.0f, 1.0f);

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
	// get normal
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	float3 out_dir = reflect(in_dir, recthelio->normal_);   // reflect light
	out_dir = normalize(out_dir);
	plane.set_pos(solar_scene->heliostats[0]->pos_, -out_dir);

	//




	// Step 3: 






}

