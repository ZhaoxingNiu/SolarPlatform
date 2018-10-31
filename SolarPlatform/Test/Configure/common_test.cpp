#include "./common_test.h"
#include <iostream>
#include <fstream>


namespace common_test {

	bool test_file_path() {
		std::ofstream f1("../SimulResult/test.txt");      //���ļ�����д�����ļ������ھʹ�����
		if (!f1) return false;                          //���ļ�ʧ�����������
		f1 << "author ��" << "nzx" << std::endl; 
		f1 << "email��" << "zhaoxingniu@163.com" << std::endl;
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