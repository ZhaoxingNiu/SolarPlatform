#include "./raytracing_test.cuh"
#include "../../Common/image_saver.h"
#include "../../RayTracing/RectHelio/recthelio_tracing.h"
#include "../../SceneProcess/PreProcess/scene_instance_process.h"

#include <sstream>

void raytracing_test(SolarScene &solar_scene)
{

	// helios
	float helio_granularity[] = { 0.01f};

	// rays
	int sun_shape_per_group[] = { 2048 };
	float csrs[] = { 0.1f};
	float disturb_stds[] = { 0.001};

	int start_n = 0, end_n = 1000;
	string save_path("../result//____.txt");
	float *h_image = nullptr;

	int index = 0;
	Receiver *recv = dynamic_cast<RectangleReceiver *>(solar_scene.receivers[0]);
	/*for (int i = 0; i < solar_scene.heliostats.size(); ++i)*/
	for (int i = 24; i < 25; ++i)
	{
		RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene.heliostats[i]);
		for (int i_gral = 0; i_gral < sizeof(helio_granularity) / sizeof(float); ++i_gral)
		{
			recthelio->pixel_length_ = helio_granularity[i_gral];
			for (int j = start_n; j < end_n; ++j)
			{
				for (int i_n_per_group = 0; i_n_per_group < sizeof(sun_shape_per_group) / sizeof(int); ++i_n_per_group)
				{
					solar_scene.sunray_->num_sunshape_lights_per_group_ = sun_shape_per_group[i_n_per_group];
					solar_scene.sunray_->CClear();
					for (int i_csr = 0; i_csr < sizeof(csrs) / sizeof(float); ++i_csr)
					{
						solar_scene.sunray_->csr_ = csrs[i_csr];
						for (int i_dist = 0; i_dist < sizeof(disturb_stds) / sizeof(float); ++i_dist)
						{
							string tmp = save_path;
							tmp.insert(tmp.size() - 9, to_string(i));
							tmp.insert(tmp.size() - 8, to_string(i_gral));
							tmp.insert(tmp.size() - 7, to_string(j));
							tmp.insert(tmp.size() - 6, to_string(i_n_per_group));
							tmp.insert(tmp.size() - 5, to_string(i_csr));
							tmp.insert(tmp.size() - 4, to_string(i_dist));

							solarenergy::disturb_std = disturb_stds[i_dist];
							// reset sunray
							SceneProcessor::set_sunray_content(*solar_scene.sunray_);

							// clean screen to all 0s
							recv->Cclean_image_content();

							// ray-tracing
							recthelio_ray_tracing(*solar_scene.sunray_,
								*recv,
								*recthelio,
								*solar_scene.grid0s[i],
								solar_scene.heliostats);

							// Save result

							global_func::gpu2cpu(h_image, recv->d_image_, recv->resolution_.x*recv->resolution_.y);
							// Id, Ssub, rou, Nc
							float Id = solar_scene.sunray_->dni_;
							float Ssub = recthelio->pixel_length_*recthelio->pixel_length_;
							float rou = solarenergy::reflected_rate;
							int Nc = solar_scene.sunray_->num_sunshape_lights_per_group_;
							float Srec = recv->pixel_length_*recv->pixel_length_;
							float max = -1.0f;
							for (int p = 0; p < recv->resolution_.x*recv->resolution_.y; ++p)
							{
								h_image[p] = h_image[p] * Id * Ssub * rou / Nc / Srec;

								if (max < h_image[p])
									max = h_image[p];
							}

							ImageSaver::savetxt(tmp, recv->resolution_.x, recv->resolution_.y, h_image);
							printf("No.%d\n", ++index);
						}
					}
				}
				printf("(%d,\t%d)\n", i, j);
			}
		}
	}
	delete[] h_image;
	h_image = nullptr;
}


void raytracing_standard_test(SolarScene &solar_scene,int hIndex,int gridIndex,string outpath) {
	RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene.heliostats[hIndex]);

	solar_scene.receivers[0]->Cclean_image_content();
	recthelio_ray_tracing(*solar_scene.sunray_,
		*solar_scene.receivers[0],
		*recthelio,
		*solar_scene.grid0s[0],
		solar_scene.heliostats);

	float *h_image = nullptr;
	global_func::gpu2cpu(h_image, solar_scene.receivers[0]->d_image_, solar_scene.receivers[0]->resolution_.x*solar_scene.receivers[0]->resolution_.y);
	// Id, Ssub, rou, Nc
	float Id = solar_scene.sunray_->dni_;
	float Ssub = recthelio->pixel_length_*recthelio->pixel_length_;
	float rou = solarenergy::reflected_rate;
	int Nc = solar_scene.sunray_->num_sunshape_lights_per_group_;
	float Srec = solar_scene.receivers[0]->pixel_length_*solar_scene.receivers[0]->pixel_length_;
	for (int i = 0; i < solar_scene.receivers[0]->resolution_.x*solar_scene.receivers[0]->resolution_.y; ++i)
	{
		h_image[i] = h_image[i] * Id * Ssub * rou / Nc / Srec;
	}
	// Save image	  "face2face_shadow-1.txt"
	ImageSaver::savetxt(outpath.c_str(), solar_scene.receivers[0]->resolution_.x, solar_scene.receivers[0]->resolution_.y, h_image);
	delete[] h_image;
	h_image = nullptr;

}