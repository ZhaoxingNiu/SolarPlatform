#ifndef DDA_STEPS_H
#define DDA_STEPS_H

#include "../../SceneProcess/solar_scene.h"
#include "../../DataStructure/heliostat.cuh"

#include "./dda_interface.h"
#include "../Struct/oblique_parallel.cuh"
#include "../Struct/convKernel.h"
#include "../Script/gen_kernel.h"
#include "../Cufft/convolutionFFT2D_interface.h"

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector>
#include <cmath>

bool set_helios_vertexes_cpu(
	const std::vector<Heliostat *> heliostats,
	const int start_pos,
	const int end_pos,
	float3 *&h_helio_vertexs);

/*
* Step 1. initialize the image plane
* Step 2. rasterization
* Step 3: init the kernel
* Step 4: convolution calculation
* Step 5: projcetion
*/
bool conv_method_kernel(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float3 normal = make_float3(0.0f,0.0f,0.0f),  // defautl do not set the normal, the image plane's normal
	kernelType k_type = kernelType::T_LOADED_CONV,
	float sigma_2 = 1.2f  // effective only k_type = kernelType::T_GAUSSIAN_CONV
);

bool conv_method_kernel_focus(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int sub_num,
	int grid_index,
	kernelType k_type = kernelType::T_LOADED_CONV,
	float sigma_2 = 1.2f  // effective only k_type = kernelType::T_GAUSSIAN_CONV
);

/*
* for the hfcal model
* the convolution calcualtion unnecessary
*/
bool conv_method_kernel_HFLCAL(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float sigma_2 = 1.2f  // effective only k_type = kernelType::T_GAUSSIAN_CONV
);


bool conv_method_kernel_HFLCAL_focus(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int sub_num,
	int grid_index,
	float sigma_2 = 1.2f  // effective only k_type = kernelType::T_GAUSSIAN_CONV
);


#endif // !DDA_STEPS_H

