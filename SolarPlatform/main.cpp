#include <iostream>
 
// 测试头文件
#include "./Test/Configure/common_test.h"
#include "./Test/Configure/test_get_file_peak.h"
#include "./Test/configure/cuda_config_test.cuh"  // check cuda configure
#include "./Test/Configure/reduce_test.cuh"

#include "./Test/Util/rapidjson_test.h"

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
	
	int nFailures = 0;
	// test basic configure
	/*
	queryDeviceInfo();
	std::string kernel_path= "../SimulResult/test.txt";
	if (!common_test::testResultPath(kernel_path)) { nFailures++; }
	if (!common_test::testFileIsExist(kernel_path)) { nFailures++; }

	// configuration
	std::string configure_file_path = "../Conf/example_configuration.json";
	if (testRapidjsonReadConf(configure_file_path)) { nFailures++; }

	// test sub function
	// if (!testGenFittedKernel(500.0f, 500.0f, 135.0f)) { nFailures++; }
	// if (!testFastConvolution()){  nFailures++; }
	// if (!testRasterizationShadowBlock()) { nFailures++;  }
	// if (!testGpuReduce()) { nFailures++; }

	*/

	if (!test_scene_format_transfer()) { nFailures++; }
	if (!test_scene_format_transfer_ps10()) { nFailures++; }
    if (!test_focus_helios_split()) { nFailures++; }

	//if (!test_raytracing()) { nFailures++; }
    //if (!test_dda_rasterization()) { nFailures++; }
	
	// modify the one point
	//if (!test_raytracing_onepoint()) { nFailures++; }
	//if (!test_gen_kernel(500.0f, 500.0f, 135.0f)) { nFailures++; }


	//***************华丽丽的分割线*****************************
	//just for paper
	//if (!test_raytracing_scene1()) { nFailures++; }
	//if (!test_conv_model_scene1()) { nFailures++; }
    //if (!test_unizar_model_scene1()) { nFailures++; }
	//if (!test_hflcal_model_scene1()) { nFailures++; }

	//if (!test_conv_model_scene_shadow()) { nFailures++; }

	//test ps 10
	//if (!test_raytracing_scene_ps10()) { nFailures++; }
	//子平面镜分开计算的版本
    //if (!test_conv_model_scene_ps10_tmp()) { nFailures++; }
	//if (!test_conv_model_scene_ps10()) { nFailures++; }
	//if (!test_unizar_model_ps10()) { nFailures++; }
	//if (!test_hflcal_model_ps10()) { nFailures++; }

	//ps 10 real
	// if (!test_conv_model_scene_ps10_real()) { nFailures++; }

 	std::cout << "nFailures number: " << nFailures << std::endl;
	system("pause");
	return 0;
}