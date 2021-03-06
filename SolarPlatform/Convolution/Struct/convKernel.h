#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

enum kernelType {T_CONV, T_LOADED_CONV, T_GAUSSIAN_CONV, T_GAUSSIAN_CONV_MATLAB,T_HFLCAL};

/*
*
* ConvKenel do not contain the ptr,just modify the kernel Value
* LoadedConvKernel is the load the kernel from the
*
*/
class ConvKernel {
public:
	ConvKernel(int h, int w);
	ConvKernel(int h, int w, std::string path);
	virtual void genKernel() = 0;
	void saveKernel(std::string path);
	virtual ~ConvKernel();

	void setSize(int h, int w);
	void getSize(int &h, int &w);
	void setModelPath(std::string path);

	int dataH;
	int dataW;
	std::string modelPath;
	float *h_data;
	float *d_data;
};

/*
* load the data array
*/
class LoadedConvKernel :public ConvKernel {
public:
	LoadedConvKernel(int h, int w, std::string path);
	virtual void genKernel();
	virtual ~LoadedConvKernel();
};


class GaussianConvKernel : public ConvKernel {
public:
	GaussianConvKernel(int h, int w, float A_, float sigma_2_, float pixel_length_ =  0.05f, float offset_ = -5.0f);
	void setKernelParam(float A_, float sigma_2_);
	virtual void genKernel();
	virtual ~GaussianConvKernel();

	float pixel_length;
	float offset;
	float A;
	float sigma_2;

};


#endif // !CONV_KERNEL_H
