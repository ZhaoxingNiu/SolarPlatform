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

}