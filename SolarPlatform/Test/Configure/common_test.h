#ifndef COMMON_TEST_H
#define COMMON_TEST_H

#include <string>

namespace common_test {

	// test the file process api
	bool testResultPath(std::string kernel_path);
	bool testFileIsExist(std::string kernel_path);

}

#endif // COMMON_TEST_H
