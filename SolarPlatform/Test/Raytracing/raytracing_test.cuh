#ifndef RAYTRACING_TEST_CUH
#define RAYTRACING_TEST_CUH

#include "../../Common/common_var.h"
#include "../../Common/random_generator.h"
#include "../../SceneProcess/solar_scene.h"

#include <iostream>

void raytracing_test(SolarScene &solar_scene);

void raytracing_standard_test(SolarScene &solar_scene,int hIndex = 0, int gridIndex = 0, string outpath = "result/test.txt");


#endif // !RAYTRACING_TEST_CUH