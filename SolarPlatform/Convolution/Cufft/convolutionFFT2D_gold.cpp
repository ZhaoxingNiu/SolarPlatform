/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "convolutionFFT2D_common.h"



 ////////////////////////////////////////////////////////////////////////////////
 // Helper functions
 ////////////////////////////////////////////////////////////////////////////////
int snapTransformSize(int dataSize)
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
		{
			break;
		}

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
	{
		return dataSize;
	}

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024)
	{
		return hiPOT;
	}
	else
	{
		return iAlignUp(dataSize, 512);
	}
}

float getRand(void)
{
	return (float)(rand() % 16);
}

void showDataConvolution(float* data, int H, int W) {
	for (int i = 0; i < H; i+=20) {
		for (int j = 0; j < W; j+=20) {
			std::cout << data[i*W + j] << " ";
		}
		std::cout << std::endl;
	}
}




////////////////////////////////////////////////////////////////////////////////
// Reference straightforward CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionClampToBorderCPU(
    float *h_Result,  //result
    float *h_Data,    //src
    float *h_Kernel,  //kernel
    int dataH,        //src's rows
    int dataW,        //src's cols
    int kernelH,      //kernel's rows
    int kernelW,      //kernel's cols
    int kernelY,      //(kernelH-1)/2
    int kernelX       //(kernelW-1)/2
)
{
    for (int y = 0; y < dataH; y++)
        for (int x = 0; x < dataW; x++)
        {
            double sum = 0;

            for (int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++)
                for (int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++)
                {
                    int dy = y + ky;
                    int dx = x + kx;

                    if (dy < 0) continue;

                    if (dx < 0) continue;

                    if (dy >= dataH) continue;

                    if (dx >= dataW) continue;

                    sum += h_Data[dy * dataW + dx] * h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)];
                }

            h_Result[y * dataW + x] = (float)sum;
        }
}
