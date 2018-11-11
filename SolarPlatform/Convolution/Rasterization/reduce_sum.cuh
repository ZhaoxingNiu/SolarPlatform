#ifndef REDUCE_SUM_CUH
#define REDUCE_SUM_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

/*
* calculate the discrete grid's area 
* the projected grid's area = num of grids the projected * pixel_length * pixel_length
*/

float get_discrete_area(float *d_data, int N, float pixel_length);



#endif // !REDUCE_SUM_CUH
