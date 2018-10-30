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

bool fastConvolution(float *h_Data, float *h_Kernel, float *h_Result,
	const int dataH, const int dataW, const int kernelH, const int kernelW, bool use_GPU = true) {
	if (use_GPU) {
		return fftConvolutionGPU(h_Data, h_Kernel, h_Result, dataH, dataW, kernelH, kernelW);
	}
	return  fftConvolutionCPU(h_Data, h_Kernel, h_Result, dataH, dataW, kernelH, kernelW);
}

bool fastConvolutionDevice(float *d_Data, float *d_Kernel,
	const int dataH, const int dataW, const int kernelH, const int kernelW) {
	return  fftConvolutionGPUDevice(d_Data, d_Kernel, dataH, dataW, kernelH, kernelW);
}

bool fftConvolutionGPU(float *h_Data, float *h_Kernel, float *h_Result,
	const int dataH, const int dataW, const int kernelH, const int kernelW)
{
	float *h_ResultOrigin;
	float *d_Data, *d_PaddedData, *d_Kernel, *d_PaddedKernel;
	fComplex *d_DataSpectrum, *d_KernelSpectrum;
	cufftHandle fftPlanFwd, fftPlanInv;

	const int kernelY = (kernelH - 1) / 2;
	const int kernelX = (kernelW - 1) / 2;

	bool bRetVal = true;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	printf("built-in R2C / C2R FFT-based convolution\n");
	const int    fftH = snapTransformSize(dataH + kernelH - 1);
	const int    fftW = snapTransformSize(dataW + kernelW - 1);

	printf("...allocating memory\n");
	h_ResultOrigin = (float *)malloc(fftH    * fftW * sizeof(float));

	checkCudaErrors(cudaMalloc((void **)&d_Data, dataH   * dataW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

	printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
	checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

	printf("...uploading to GPU and padding convolution kernel and input data\n");
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, dataH   * dataW * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	padKernel(
		d_PaddedKernel,
		d_Kernel,
		fftH,
		fftW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);
#ifdef CONV_DEBUG
	float *h_PaddedKernel = (float *)malloc(fftH    * fftW * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_PaddedKernel, d_PaddedKernel, fftH    * fftW * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "h_Kernel" << std::endl;
	showDataConvolution(h_PaddedKernel, fftH, fftW);
#endif

	padDataClampToBorder(
		d_PaddedData,
		d_Data,
		fftH,
		fftW,
		dataH,
		dataW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);
#ifdef CONV_DEBUG
	float *h_PaddedData = (float *)malloc(fftH    * fftW * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_PaddedData, d_PaddedData, fftH    * fftW * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "h_PaddedData" << std::endl;
	showDataConvolution(h_PaddedData, fftH, fftW);
#endif

	//Not including kernel transformation into time measurement,
	//since convolution kernel is not changed very frequently
	printf("...transforming convolution kernel\n");
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));

	printf("...running GPU FFT convolution: ");
	checkCudaErrors(cudaDeviceSynchronize());
	//sdkResetTimer(&hTimer);
	//sdkStartTimer(&hTimer);
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
	modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
	checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

	printf("...reading back GPU convolution results\n");
	checkCudaErrors(cudaMemcpy(h_ResultOrigin, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
	
	printf("...reading back origin data to the result\n");
	for (int y = 0; y < dataH; y++) {
		for (int x = 0; x < dataW; x++)
		{
			h_Result[y * dataW + x] = h_ResultOrigin[y * fftW + x];
		}
	}

	checkCudaErrors(cufftDestroy(fftPlanInv));
	checkCudaErrors(cufftDestroy(fftPlanFwd));

	checkCudaErrors(cudaFree(d_DataSpectrum));
	checkCudaErrors(cudaFree(d_KernelSpectrum));
	checkCudaErrors(cudaFree(d_PaddedData));
	checkCudaErrors(cudaFree(d_PaddedKernel));
	checkCudaErrors(cudaFree(d_Data));
	checkCudaErrors(cudaFree(d_Kernel));
	free(h_ResultOrigin);

	return bRetVal;
}

bool fftConvolutionGPUDevice(float *d_Data, float *d_Kernel,
	const int dataH, const int dataW, const int kernelH, const int kernelW)
{
	float *h_ResultOrigin;
	float *d_PaddedData, *d_PaddedKernel;
	fComplex *d_DataSpectrum, *d_KernelSpectrum;
	cufftHandle fftPlanFwd, fftPlanInv;

	const int kernelY = (kernelH - 1) / 2;
	const int kernelX = (kernelW - 1) / 2;

	bool bRetVal = true;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	printf("built-in R2C / C2R FFT-based convolution\n");
	const int    fftH = snapTransformSize(dataH + kernelH - 1);
	const int    fftW = snapTransformSize(dataW + kernelW - 1);

	printf("...allocating memory\n");
	float *h_Result = (float *)malloc(dataH * dataW * sizeof(float));
	h_ResultOrigin = (float *)malloc(fftH    * fftW * sizeof(float));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

	printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
	checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

	printf("...uploading to GPU and padding convolution kernel and input data\n");
	checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	padKernel(
		d_PaddedKernel,
		d_Kernel,
		fftH,
		fftW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);
#ifdef CONV_DEBUG
	float *h_PaddedKernel = (float *)malloc(fftH    * fftW * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_PaddedKernel, d_PaddedKernel, fftH    * fftW * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "h_Kernel" << std::endl;
	showDataConvolution(h_PaddedKernel, fftH, fftW);
#endif

	padDataClampToBorder(
		d_PaddedData,
		d_Data,
		fftH,
		fftW,
		dataH,
		dataW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);
#ifdef CONV_DEBUG
	float *h_PaddedData = (float *)malloc(fftH    * fftW * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_PaddedData, d_PaddedData, fftH    * fftW * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "h_PaddedData" << std::endl;
	showDataConvolution(h_PaddedData, fftH, fftW);
#endif

	//Not including kernel transformation into time measurement,
	//since convolution kernel is not changed very frequently
	printf("...transforming convolution kernel\n");
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));

	printf("...running GPU FFT convolution: ");
	checkCudaErrors(cudaDeviceSynchronize());
	//sdkResetTimer(&hTimer);
	//sdkStartTimer(&hTimer);
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
	modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
	checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

	printf("...reading back GPU convolution results\n");
	checkCudaErrors(cudaMemcpy(h_ResultOrigin, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));

	printf("...reading back origin data to the result\n");
	for (int y = 0; y < dataH; y++) {
		for (int x = 0; x < dataW; x++)
		{
			h_Result[y * dataW + x] = h_ResultOrigin[y * fftW + x];
		}
	}

#ifdef CONV_DEBUG
	std::cout << "h_Result in function" << std::endl;
	showDataConvolution(h_Result, dataH, dataW);
#endif

	// process the data
	checkCudaErrors(cudaMemcpy(d_Data, h_Result, dataH * dataW * sizeof(float),
		cudaMemcpyHostToDevice));

	checkCudaErrors(cufftDestroy(fftPlanInv));
	checkCudaErrors(cufftDestroy(fftPlanFwd));

	checkCudaErrors(cudaFree(d_DataSpectrum));
	checkCudaErrors(cudaFree(d_KernelSpectrum));
	checkCudaErrors(cudaFree(d_PaddedData));
	checkCudaErrors(cudaFree(d_PaddedKernel));
	free(h_ResultOrigin);
	return bRetVal;
}

bool fftConvolutionCPU(float *h_Data, float *h_Kernel, float *h_Result,
	const int dataH, const int dataW, const int kernelH, const int kernelW)
{
	printf("CPU convolution\n");
	bool bRetVal = true;
	StopWatchInterface *hTimer = NULL;
	const int kernelY = (kernelH - 1) / 2;
	const int kernelX = (kernelW - 1) / 2;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	convolutionClampToBorderCPU(
		h_Result,
		h_Data,
		h_Kernel,
		dataH,
		dataW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);
	sdkStopTimer(&hTimer);
	double gpuTimeCPU = sdkGetTimerValue(&hTimer);
	printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTimeCPU * 0.001), gpuTimeCPU);
	return bRetVal;
}


bool checkConvolution(float *h_ResultGPU, float *h_ResultCPU, int dataH, int dataW) {
	printf("comparing the results: ");
	bool bRetVal = true;
	double sum_delta2 = 0;
	double sum_ref2 = 0;
	double max_delta_ref = 0;

	for (int y = 0; y < dataH; y++)
		for (int x = 0; x < dataW; x++)
		{
			double  rCPU = (double)h_ResultCPU[y * dataW + x];
			double  rGPU = (double)h_ResultGPU[y * dataW + x];
			double delta = (rCPU - rGPU) * (rCPU - rGPU);
			double   ref = rCPU * rCPU + rCPU * rCPU;

			if ((delta / ref) > max_delta_ref)
			{
				max_delta_ref = delta / ref;
			}

			sum_delta2 += delta;
			sum_ref2 += ref;
		}

	double L2norm = sqrt(sum_delta2 / sum_ref2);
	printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
	bRetVal = (L2norm < 1e-6) ? true : false;
	printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");
	return bRetVal;
}