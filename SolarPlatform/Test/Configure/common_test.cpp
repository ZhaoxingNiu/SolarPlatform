#include "./common_test.h"
#include <iostream>
#include <fstream>

namespace common_test {

	bool testResultPath(std::string kernel_path) {
		std::ofstream f1(kernel_path);                  //���ļ�����д�����ļ������ھʹ���
		if (!f1) return false;                          //���ļ�ʧ�����������
		f1 << "here is the result" << std::endl; 
		f1.close();
		return true;
	}

	bool testFileIsExist(std::string kernel_path) {
		std::ifstream fin(kernel_path);
		if (!fin) {
			std::cout << " file do not exist " << std::endl;
			return false;
		}
		return true;
	}
}