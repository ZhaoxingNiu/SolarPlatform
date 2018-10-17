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

}