#ifndef GEN_KERNEL_TEST_H
#define GEN_KERNEL_TEST_H

#include "../../Convolution/Script/gen_fitted_kernel.h"
#include "../../Convolution/Struct/convKernel.h"

// 测试调用py生成对应文件
bool testGenFittedKernel(float ori_dis, float true_dis, float angel);

bool testLoadKernel();

#endif // !GEN_KERNEL_TEST_H

