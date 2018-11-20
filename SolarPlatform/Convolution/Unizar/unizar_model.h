#ifndef DDA_TEST_H

#include "../3DDDA/dda_interface.h"
#include "../3DDDA/dda_steps.h"
#include "../../SceneProcess/PreProcess/scene_instance_process.h"

void unizar_model(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float ideal_peak,
	std::string resa_path
);

bool test_unizar_model();


#endif // !DDA_TEST_H