#include "convolutionFFT2D_interface.h"
#include "convolutionFFT2D_common.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

bool testFastConvolution(void) {
	bool bRetVal = true;
	const int kernelH = 201;
	const int kernelW = 201;
	const int   dataH = 201;
	const int   dataW = 201;

	float *h_Data, *h_Kernel, *h_ResultGPU, *h_ResultCPU;
	h_Data = (float *)malloc(dataH   * dataW * sizeof(float));
	h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
	h_ResultGPU = (float *)malloc(dataH    * dataW * sizeof(float));
	h_ResultCPU = (float *)malloc(dataH    * dataW * sizeof(float));

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
#ifdef CONV_DEBUG
	std::cout << "h_Data" << std::endl;
	showDataConvolution(h_Data, dataH, dataW);
	std::cout << "h_Kernel" << std::endl;
	showDataConvolution(h_Kernel, kernelH, kernelW);
#endif
	fftConvolutionGPU(h_Data, h_Kernel, h_ResultGPU, dataH, dataW, kernelH, kernelW);
	fftConvolutionCPU(h_Data, h_Kernel, h_ResultCPU, dataH, dataW, kernelH, kernelW);

#ifdef CONV_DEBUG
	std::cout << "h_ResultGPU" << std::endl;
	showDataConvolution(h_ResultGPU, dataH, dataW);
	std::cout << "h_ResultCPU" << std::endl;
	showDataConvolution(h_ResultCPU, dataH, dataW);
#endif


	bRetVal = checkConvolution(h_ResultGPU, h_ResultCPU, dataH, dataW);
	free(h_ResultGPU);
	free(h_ResultCPU);
	free(h_Data);
	free(h_Kernel);
	return bRetVal;
}