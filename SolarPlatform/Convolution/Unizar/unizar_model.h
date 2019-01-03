#ifndef UNIZAR_MODEL_H
#define UNIZAR_MODEL_H


#include "../3DDDA/dda_interface.h"
#include "../3DDDA/dda_steps.h"
#include "../../SceneProcess/PreProcess/scene_instance_process.h"

void unizar_model(
	SolarScene *solar_scene,
	AnalyticModelScene *model_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float ideal_peak,
	std::string res_path,
	bool is_focus = false,
	int sub_num = 1

);

bool test_unizar_model_scene1();

bool test_unizar_model_ps10();

#endif // !UNIZAR_MODEL_H