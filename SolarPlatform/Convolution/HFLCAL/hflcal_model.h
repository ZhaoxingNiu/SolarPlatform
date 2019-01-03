#ifndef HFLCAL_MODEL_H
#define HFLCAL_MODEL_H

#include "../3DDDA/dda_interface.h"
#include "../3DDDA/dda_steps.h"
#include "../../SceneProcess/PreProcess/scene_instance_process.h"

void hflcal_model(
	SolarScene *solar_scene,
	AnalyticModelScene *model_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float ideal_peak,
	std::string resa_path
);

void hflcal_model_focus(
	SolarScene *solar_scene,
	AnalyticModelScene *model_scene,
	int rece_index,
	int helio_index,
	int sub_num,
	int grid_index,
	float ideal_peak,
	std::string resa_path
);

bool test_hflcal_model_scene1();

bool test_hflcal_model_ps10();


#endif //HFLCAL_MODEL_H