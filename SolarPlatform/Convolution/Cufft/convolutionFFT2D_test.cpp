#include "convolutionFFT2D_interface.h"
#include "convolutionFFT2D_common.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

bool testFastConvolution(void) {
	bool bRetVal = true;
	const int kernelH = 3;
	const int kernelW = 3;
	const int   dataH = 11;
	const int   dataW = 11;

	float *h_Data, *h_Kernel, *h_ResultGPU, *h_ResultCPU, *h_ResultDevice;
	h_Data = (float *)malloc(dataH   * dataW * sizeof(float));
	h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
	h_ResultGPU = (float *)malloc(dataH    * dataW * sizeof(float));
	h_ResultCPU = (float *)malloc(dataH    * dataW * sizeof(float));
	h_ResultDevice = (float *)malloc(dataH    * dataW * sizeof(float));

    // test the device function
	float *d_Data, *d_Kernel;
	checkCudaErrors(cudaMalloc((void **)&d_Data, dataH   * dataW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

	printf("...generating random input data\n");
	srand(2010);
	for (int i = 0; i < dataH * dataW; i++)
	{
		h_Data[i] = 1;
	}
	for (int i = 0; i < kernelH * kernelW; i++)
	{
		h_Kernel[i] = 1;
	}

	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, dataH   * dataW * sizeof(float), cudaMemcpyHostToDevice));

#ifdef CONV_DEBUG
	std::cout << "h_Data" << std::endl;
	showDataConvolution(h_Data, dataH, dataW);
	std::cout << "h_Kernel" << std::endl;
	showDataConvolution(h_Kernel, kernelH, kernelW);
#endif
	fftConvolutionGPU(h_Data, h_Kernel, h_ResultGPU, dataH, dataW, kernelH, kernelW);
	fftConvolutionCPU(h_Data, h_Kernel, h_ResultCPU, dataH, dataW, kernelH, kernelW);
	
	// device api
	fastConvolutionDevice(d_Data, d_Kernel, dataH, dataW, kernelH, kernelW);
	checkCudaErrors(cudaMemcpy(h_ResultDevice, d_Data, dataH   * dataW * sizeof(float), cudaMemcpyDeviceToHost));
#define CONV_DEBUG
#ifdef CONV_DEBUG
	std::cout << "h_ResultGPU" << std::endl;
	showDataConvolution(h_ResultGPU, dataH, dataW);
	std::cout << "h_ResultCPU" << std::endl;
	showDataConvolution(h_ResultCPU, dataH, dataW);
	std::cout << "h_ResultDevice" << std::endl;
	showDataConvolution(h_ResultDevice, dataH, dataW);
#endif

	bRetVal = checkConvolution(h_ResultGPU, h_ResultCPU, dataH, dataW);
	//bRetVal = checkConvolution(h_ResultGPU, h_ResultDevice, dataH, dataW);

	checkCudaErrors(cudaFree(d_Data));
	checkCudaErrors(cudaFree(d_Kernel));
	free(h_ResultGPU);
	free(h_ResultCPU);
	free(h_Data);
	free(h_Kernel);
	return bRetVal;
}