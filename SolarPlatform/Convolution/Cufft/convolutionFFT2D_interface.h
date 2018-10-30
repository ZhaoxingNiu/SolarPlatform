#ifndef CONVOLUTION_INTERFACE_H
#define CONVOLUTION_INTERFACE_H

////////////////////////////////
// convolution interface
////////////////////////////////
bool fastConvolutionDevice(
	float *d_Data,
	float *d_Kernel,
	const int dataH,
	const int dataW,
	const int kernelH,
	const int kernelW
);

//////////////////////////////
//  calcluate convolution on the GPU
/////////////////////////////
bool fftConvolutionGPUDevice(float *d_Data, float *d_Kernel, 
	const int dataH, const int dataW, const int kernelH, const int kernelW);


////////////////////////////////
// convolution interface
////////////////////////////////
bool fastConvolution(
	float *h_Data,    
	float *h_Kernel, 
	float *h_Result,
	const int dataH, 
	const int dataW, 
	const int kernelH, 
	const int kernelW, 
	bool use_GPU  // defaule: true, using GPU to compute the data
);

///////////////////////////////
// calculate convolution on the CPU
///////////////////////////////
bool fftConvolutionCPU(float *h_Data, float *h_Kernel, float *h_Result,
	const int dataH, const int dataW, const int kernelH, const int kernelW);

//////////////////////////////
//  calcluate convolution on the GPU
/////////////////////////////
bool fftConvolutionGPU(float *h_Data, float *h_Kernel, float *h_Result,
	const int dataH, const int dataW, const int kernelH, const int kernelW);


/////////////////////////////
//  checkout the GPU value
/////////////////////////////
bool checkConvolution(float *h_ResultGPU, float *h_ResultCPU, int dataH, int dataW);

#endif // !CONVOLUTION_INTERFACE_H