#include "./ConvKernel.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>

ConvKernel::ConvKernel(int h, int w) :dataH(h), dataW(w) {

}

ConvKernel::ConvKernel(int h, int w, std::string path) : dataH(h), dataW(w), modelPath(path) {

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

void ConvKernel::Conv




///////////////
//  load the data array
///////////////

LoadedConvKernel::LoadedConvKernel(int h, int w, std::string path) :ConvKernel(h, w, path) {

}

void LoadedConvKernel::genKernel(float* h_Kernel) {
	std::ifstream infile;
	infile.open(modelPath);   //���ļ����������ļ��������� 
	assert(infile.is_open());   //��ʧ��,�����������Ϣ,����ֹ�������� 

	std::string s;
	std::stringstream ss;
	int r = 0;
	while (getline(infile, s))
	{
		ss << s;
		int c = 0;
		float num;
		while (ss >> num) {
			h_Kernel[r * dataW + c] = num;
			++c;
		}
		++r;
	}
	infile.close();             //�ر��ļ������� 
}