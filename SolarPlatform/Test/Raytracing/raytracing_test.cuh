#ifndef RAYTRACING_TEST_CUH
#define RAYTRACING_TEST_CUH

#include "../../Common/common_var.h"
#include "../../Common/random_generator.h"
#include "../../SceneProcess/solar_scene.h"

#include <iostream>

void raytracing_interface(SolarScene &solar_scene);

void raytracing_standard_interface(SolarScene &solar_scene,int hIndex = 0, int gridIndex = 0, string outpath = "result/test.txt");

bool test_raytracing();
bool test_raytracing_onepoint();

// just for paper
bool test_raytracing_scene1();
bool test_raytracing_scene_ps10();

#endif // !RAYTRACING_TEST_CUH