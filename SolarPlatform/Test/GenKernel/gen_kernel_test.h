#ifndef GEN_KERNEL_TEST_H
#define GEN_KERNEL_TEST_H

#include "../../Convolution/Script/gen_kernel.h"
#include "../../Convolution/Struct/convKernel.h"

bool test_gen_kernel(float ori_dis, float true_dis, float angel);

bool test_gen_kernel_gaussian(float ori_dis, float true_dis, float angel);

bool test_load_kernel();

#endif // !GEN_KERNEL_TEST_H

