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


bool conv_method_kernel(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	kernelType k_type = kernelType::T_LOADED_CONV,
	float sigma_2 = 1.2f
);

#endif // !DDA_STEPS_H

