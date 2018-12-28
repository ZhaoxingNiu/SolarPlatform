#include <iostream>
 
#include "./Test/Configure/common_test.h"
#include "./Test/Configure/test_get_file_peak.h"
#include "./Test/configure/cuda_config_test.cuh"  // check cuda configure
#include "./Test/Configure/reduce_test.cuh"


#include "./Test/Raytracing/raytracing_test.cuh"
#include "./Test/GenKernel/gen_kernel_test.h"
#include "./Test/SceneTrans/FormatTransfer.h"
#include "./Test/SceneTrans/FocusHeliosSplits.h"


#include "./Convolution/Cufft/convolutionFFT2D_test.h"
#include "./Convolution/Rasterization/rasterization_test.h"
#include "./Convolution/3DDDA/dda_test.h"
#include "./Convolution/model/conv_model.h"
#include "./Convolution/Unizar/unizar_model.h"
#include "./Convolution/HFLCAL/hflcal_model.h"



int main() {
	
	// test cuda's configure
	//std::cout << "test cuda configure" << std::endl;
	//test_cuda_conf();
	
    int nFailures = 0;
	// if (!common_test::test_file_path()) { nFailures++; }
	// if (!common_test::test_file_exist()) { nFailures++; }

	// if (!test_gen_kernel(500.0f, 500.0f, 135.0f)) { nFailures++; }
	// if (!test_gen_kernel_gaussian(500.0f, 500.0f, 135.0f)) { nFailures++; }
	// if (!testFastConvolution()){  nFailures++; }
	// if (!test_rasterization()) { nFailures++;  }
	// if (!test_load_kernel()) { nFailures++;}
	// if (!test_reduce()) { nFailures++; }
	// if (!test_scene_format_transfer()) { nFailures++; }
	// if (!test_scene_format_transfer_ps10()) { nFailures++; }
	if (!test_focus_helios_split()) { nFailures++; }



	//if (!test_raytracing()) { nFailures++; }

    //if (!test_dda_rasterization()) { nFailures++; }
	//if (!test_unizar_model()) { nFailures++; }
	//if (!test_hflcal_model()) { nFailures++; }

	//just for paper
	//if (!test_raytracing_scene1()) { nFailures++; }
	//if (!test_conv_model_scene1()) { nFailures++; }
    //if (!test_unizar_model_scene1()) { nFailures++; }
	//if (!test_hflcal_model_scene1()) { nFailures++; }



 	std::cout << "nFailures number: " << nFailures << std::endl;
	system("pause");
	return 0;
}