#include "./rasterization_test.h"
#include <cuda_runtime.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>

bool testRasterizationShadowBlock(void) {
	int rows = 100;
	int cols = 100;
	float pixel_length = 0.1;

	ProjectionPlane plane(rows, cols, pixel_length);

	// set data 
	std::vector<float3> vec1;
	vec1.push_back(make_float3(-5.0f, -5.0f, 0.0f));
	vec1.push_back(make_float3(5.0f, -5.0f, 0.0f));
	vec1.push_back(make_float3(5.0f, 5.0f, 0.0f));
	vec1.push_back(make_float3(-5.0f, 5.0f, 0.0f));

	std::vector<std::vector<float3>> vevec2;
	std::vector<float3> vec2;
	vec2.push_back(make_float3(1.0f, 1.0f, 0.0f));
	vec2.push_back(make_float3(4.0f, 1.0f, 0.0f));
	vec2.push_back(make_float3(4.0f, 4.0f, 0.0f));
	vec2.push_back(make_float3(1.0f, 4.0f, 0.0f));
	vevec2.push_back(vec2);

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// time counter
	// projection
	plane.projection(vec1);
	// shadow & block
	plane.shadow_block(vevec2);

	sdkStopTimer(&hTimer);
	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("%f MPix/s (%f ms)\n", (double)rows * (double)cols * 1e-6 / (gpuTime * 0.001), gpuTime);


	// show data 
#ifdef _DEBUG
	std::string rasterization_path = "../SimulResult/imageplane/rasterization_path.txt";
	plane.save_data_text(rasterization_path);
#endif

	return true;
}