#include "./rasterization_test.h"
#include <cuda_runtime.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>

bool test_rasterization(void) {
	int rows = 1000;
	int cols = 1000;
	float pixel_length = 0.01;
	float row_offset = -(rows - 1) * pixel_length / 2;
	float col_offset = -(cols - 1) * pixel_length / 2;
	ProjectionPlane plane(rows, cols, pixel_length, row_offset, col_offset);

	// alloc data 
	float *d_Data;
	checkCudaErrors(cudaMalloc((void **)&d_Data, rows * cols * sizeof(float)));
	checkCudaErrors(cudaMemset(d_Data, 0.0, rows * cols * sizeof(float)));
	plane.setDeviceData(d_Data);


	// set data 
	std::vector<float3> vec1;
	vec1.push_back(make_float3(-2.0f, -2.0f, 0.0f));
	vec1.push_back(make_float3(2.0f, -2.0f, 0.0f));
	vec1.push_back(make_float3(2.0f, 2.0f, 0.0f));
	vec1.push_back(make_float3(-2.0f, 2.0f, 0.0f));

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
	plane.shadowBlock(vevec2);

	sdkStopTimer(&hTimer);
	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("%f MPix/s (%f ms)\n", (double)rows * (double)cols * 1e-6 / (gpuTime * 0.001), gpuTime);


	// show data 

	float *h_Data = (float *)malloc(rows * cols * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_Data, d_Data, rows * cols * sizeof(float),
		cudaMemcpyDeviceToHost));
	for (int i = 0; i < rows; i+=100) {
		for (int j = 0; j < cols; j+=100) {
			printf("%.2f ", h_Data[i*cols + j]);
		}
		printf("\n");
	}

	// free the data
	checkCudaErrors(cudaFree(d_Data));
	free(h_Data);
	return true;
}