#include "./common_test.h"
#include <iostream>
#include <fstream>


namespace common_test {

	bool test_file_path() {
		std::ofstream f1("../SimulResult/test.txt");      //打开文件用于写，若文件不存在就创建它
		if (!f1) return false;                          //打开文件失败则结束运行
		f1 << "author ：" << "nzx" << std::endl; 
		f1 << "email：" << "zhaoxingniu@163.com" << std::endl;
		f1.close();
		return true;
	}


	bool test_file_exist() {
		std::string kernel_path = "../SimulResult/data/gen_flux/onepoint_angle_0_distance_500.txt";
		std::ifstream fin(kernel_path);
		if (!fin) {
			std::cout << " file do not exist " << std::endl;
		}

		std::string kernel_path2 = "../SimulResult/data/gen_flux/onepoint_angle_135_distance_500.txt";
		std::ifstream fin2(kernel_path2);
		if (!fin2) {
			std::cout << " file2 do not exist " << std::endl;
		}
		return true;
	}
}