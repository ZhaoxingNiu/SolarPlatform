#include "./common_test.h"
#include <iostream>
#include <fstream>

namespace common_test {

	bool testResultPath(std::string kernel_path) {
		std::ofstream f1(kernel_path);                  //打开文件用于写，若文件不存在就创建
		if (!f1) return false;                          //打开文件失败则结束运行
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