#include "./ConvKernel.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>

#include "../../Common/image_saver.h"
#include "../../Common/utils.h"
#include "../../Common/global_function.cuh"

ConvKernel::ConvKernel(int h, int w):
	dataH(h), dataW(w) {

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

	std::string s;
	std::stringstream ss;
	int r = 0;
	while (getline(infile, s))
	{
		ss << s;
		int c = 0;
		float num;
		while (ss >> num) {
			h_data[r * dataW + c] = num;
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

// Gausssian Conv kernel is implied
void GaussianConvKernel::genKernel() {
	// TODO
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