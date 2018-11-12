#include "./ConvKernel.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>

#include "../../Common/image_saver.h"
#include "../../Common/utils.h"
#include "../../Common/global_function.cuh"
#include "../../Common/global_constant.h"
#include "../../Common/common_var.h"

ConvKernel::ConvKernel(int h, int w):
	dataH(h), dataW(w), h_data(nullptr), d_data(nullptr) {
}

ConvKernel::ConvKernel(int h, int w, std::string path): 
	dataH(h), dataW(w), modelPath(path),h_data(nullptr),d_data(nullptr) {
}

void ConvKernel::setSize(int h, int w) {
	dataH = h;
	dataW = w;
}

void ConvKernel::getSize(int &h, int &w) {
	h = dataH;
	w = dataW;
}

void ConvKernel::setModelPath(std::string path) {
	modelPath = path;
}

void ConvKernel::saveKernel(std::string path) {

	// save the txt
	ImageSaver::savetxt_conv(
		path,
		dataH,
		dataW,
		h_data
	);
}

ConvKernel::~ConvKernel() {
	if (h_data) {
		delete[] h_data;
		h_data = nullptr;

	}
	if (d_data) {
		checkCudaErrors(cudaFree(d_data));
		d_data = nullptr;
	}
}


///////////////
//  load the data array
///////////////

LoadedConvKernel::LoadedConvKernel(int h, int w, std::string path):
	ConvKernel(h, w, path) {

}

void LoadedConvKernel::genKernel() {
	std::ifstream infile;
	infile.open(modelPath);   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	// clear the pointer
	if (h_data) {
		delete[] h_data;
		h_data = nullptr;
	}
	if (d_data) {
		checkCudaErrors(cudaFree(d_data));
		d_data = nullptr;
	}

	float area = solarenergy::image_plane_pixel_length * solarenergy::image_plane_pixel_length;
	h_data = new float[dataH*dataW];
	std::string s;
	std::stringstream ss;
	int r = 0;
	while (getline(infile, s))
	{
		ss.clear();
		ss << s;
		int c = 0;
		float num;
		while (ss >> num) {
			h_data[r * dataW + c] = num * area;
			++c;
		}
		++r;
	}
	
	infile.close();             //关闭文件输入流 
	//sync the d_data
	global_func::cpu2gpu(d_data, h_data, dataH * dataW);
}

LoadedConvKernel::~LoadedConvKernel() {
	if (h_data) {
		delete[] h_data;
		h_data = nullptr;

	}
	if (d_data) {
		checkCudaErrors(cudaFree(d_data));
		d_data = nullptr;
	}
}


GaussianConvKernel::GaussianConvKernel(
	int h, int w, float A_, float sigma_2_,
	float pixel_length_, float offset_):
	ConvKernel(h,w),A(A_),sigma_2(sigma_2_),pixel_length(pixel_length_),offset(offset_){
	
}

void GaussianConvKernel::setKernelParam(float A_, float sigma_2_) {
	A = A_;
	sigma_2 = sigma_2_;
}

inline float my_gaussian(float x_pos, float y_pos, float A, float sigma_2){
	return A / 2.0 / MATH_PI / sigma_2 * expf( -(x_pos * x_pos + y_pos * y_pos)/2/sigma_2);
}

// Gausssian Conv kernel is implied
void GaussianConvKernel::genKernel() {
	if (h_data) {
		delete[] h_data;
		h_data = nullptr;
	}
	if (d_data) {
		checkCudaErrors(cudaFree(d_data));
		d_data = nullptr;
	}
	float area = solarenergy::image_plane_pixel_length * solarenergy::image_plane_pixel_length;
	h_data = new float[dataH*dataW];

	for (int x = 0; x <= 200; ++x) {
		for (int y = 0; y <= 200; ++y) {
			float x_pos = offset + x * pixel_length;
			float y_pos = offset + y * pixel_length;
			h_data[x * dataW +y] = my_gaussian(x_pos, y_pos, A, sigma_2)*area;
		}
	}

	//sync the d_data
	global_func::cpu2gpu(d_data, h_data, dataH * dataW);
}

GaussianConvKernel::~GaussianConvKernel() {
	if (h_data) {
		delete[] h_data;
		h_data = nullptr;
	}
	if (d_data) {
		checkCudaErrors(cudaFree(d_data));
		d_data = nullptr;
	}
}