#ifndef GEN_KERNEL_TEST_H
#define GEN_KERNEL_TEST_H

#include "../../Convolution/Script/gen_fitted_kernel.h"
#include "../../Convolution/Struct/convKernel.h"

// ���Ե���py���ɶ�Ӧ�ļ�
bool testGenFittedKernel(float ori_dis, float true_dis, float angel);

bool testLoadKernel();

#endif // !GEN_KERNEL_TEST_H

