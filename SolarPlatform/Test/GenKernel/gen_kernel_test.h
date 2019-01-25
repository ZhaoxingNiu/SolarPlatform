#ifndef GEN_KERNEL_TEST_H
#define GEN_KERNEL_TEST_H

#include "../../Convolution/Script/gen_kernel.h"
#include "../../Convolution/Struct/convKernel.h"
#include "../../Convolution/GenKernel/KernelManager.h"

bool test_gen_kernel(float ori_dis, float true_dis, float angel);

// 调用matlab的程序生成高斯核函数，depression 
bool test_gen_kernel_gaussian(float ori_dis, float true_dis, float angel);

bool test_load_kernel();

// test the KernelManager
bool test_kernel_manager();

#endif // !GEN_KERNEL_TEST_H

