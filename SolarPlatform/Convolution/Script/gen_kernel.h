#ifndef GEN_KERNEL_H
#define GEN_KERNEL_H

void gen_kernel(
	float distance,
	float angel = 0.0f,
	float step_r = 0.05f,
	float grid_len = 0.05f,
	float distance_threshold = 0.1f,
	float rece_width = 10.05f,
	float rece_height = 10.05f,
	float rece_max_r = 7.0f);


#endif // !GEN_KERNEL_H
