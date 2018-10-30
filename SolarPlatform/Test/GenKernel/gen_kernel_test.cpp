#include "./gen_kernel_test.h"

bool test_gen_kernel() {
	bool ret = false;
	gen_kernel(
		500.0f,
		0.0f
	);
	ret = true;
	return ret;
}