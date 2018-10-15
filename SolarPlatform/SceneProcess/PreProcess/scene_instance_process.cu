#include "./scene_instance_process.h"
#include <iostream>
#include <fstream>

// sunray
void SceneProcessor::set_sunray_content(SunRay &sunray)
{
	sunray.sun_dir_ = normalize(sunray.sun_dir_);
	set_perturbation(sunray);
	set_samplelights(sunray);
}

// perturbation
__global__ void map_turbulance(float3 *d_turbulance, const float *d_guassian, const float *d_uniform, const size_t size)
{	
	unsigned int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	float theta = d_guassian[myId], phi = d_uniform[myId] * 2 * MATH_PI;
	float3 dir = global_func::angle2xyz(make_float2(theta, phi));
	d_turbulance[myId] = dir;
}

void SceneProcessor::set_perturbation(SunRay &sunray)
{
	int size = sunray.num_sunshape_lights_per_group_;
	//	Step 1:	Allocate memory for sunray.d_perturbation_ on GPU
	if (sunray.d_perturbation_ == nullptr)
		checkCudaErrors(cudaMalloc((void **)&sunray.d_perturbation_, sizeof(float3)*size));

	//	Step 2:	Allocate memory for theta and phi
	float *d_guassian_theta = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_guassian_theta, sizeof(float)*size));
	float *d_uniform_phi = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_uniform_phi, sizeof(float)*size));

	//	Step 3:	Generate theta and phi
	RandomGenerator::gpu_Gaussian(d_guassian_theta, 0.0f, solarenergy::disturb_std, size);
	RandomGenerator::gpu_Uniform(d_uniform_phi, size);

	//	Step 4:	(theta, phi) -> ( x, y, z)
	int nThreads;
	dim3 nBlocks;
	global_func::setThreadsBlocks(nBlocks, nThreads, size);
	map_turbulance << <nBlocks, nThreads >> > (sunray.d_perturbation_, d_guassian_theta, d_uniform_phi, size);

	//	Step 5: Cleanup
	checkCudaErrors(cudaFree(d_guassian_theta));
	checkCudaErrors(cudaFree(d_uniform_phi));
}

// samplelights
float sunshape_normalizedintensity(const float &theta, const float &CSR)
{
	if (theta > 0 && theta <= 4.65)
		return cosf(0.326*theta) / cosf(0.308*theta);
	else if (theta > 4.65)
	{
		float k = 0.9*logf(13.5*CSR)*powf(CSR, -0.3);
		float gamma = 2.2*logf(0.52*CSR)*powf(CSR, 0.43) - 0.1;
		return exp(k)*pow(theta, gamma);
	}
	return 0.0;
}

float sunshape_normalizedintensity(const float &theta, const float &k, const float &gamma)
{
	if (theta > 0 && theta <= 4.65)
		return cosf(0.326*theta) / cosf(0.308*theta);
	else if (theta > 4.65)
		return exp(k)*pow(theta, gamma);

	return 0.0;
}

void SceneProcessor::set_samplelights(SunRay &sunray)
{
	int num_all_lights = sunray.num_sunshape_groups_ * sunray.num_sunshape_lights_per_group_;
	int lights_3groups = sunray.num_sunshape_lights_per_group_ * 3;
	float CSR = sunray.csr_;
	float k = 0.9*logf(13.5*CSR)*powf(CSR, -0.3);
	float gamma = 2.2*logf(0.52*CSR)*powf(CSR, 0.43) - 0.1;

	//	Step 1:	Allocate memory temporarily used as h_samplelights, h_tmp_theta and h_tmp_phi on CPU
	float3 *h_samplelights = new float3[num_all_lights];
	float *h_tmp_theta = new float[lights_3groups];
	float *h_tmp_phi = new float[lights_3groups];
	float *Intens = new float[lights_3groups];

	//	Step 2:
	for (int g = 0; g < sunray.num_sunshape_groups_; ++g)
	{
		RandomGenerator::cpu_Uniform(h_tmp_phi, lights_3groups);
		RandomGenerator::cpu_Uniform(h_tmp_theta, lights_3groups);

		float maxValue = -INT_MAX, minValue = INT_MAX;
		for (int i = 0; i < lights_3groups; ++i)
		{
			Intens[i] = sunshape_normalizedintensity(h_tmp_theta[i] * 9.3f, k, gamma);

			maxValue = (Intens[i] > maxValue) ? Intens[i] : maxValue;
			minValue = (Intens[i] < minValue) ? Intens[i] : minValue;

			h_tmp_theta[i] = h_tmp_theta[i] * 9.3f / 1000.0f;
			h_tmp_phi[i] *= 2 * MATH_PI;
		}

		float range = maxValue - minValue;
		int size = 0;
		for (int i = 0; i < lights_3groups && size<sunray.num_sunshape_lights_per_group_; ++i)
		{
			float mark = float(i+1) / float(lights_3groups);
			if ((Intens[i] - minValue) / range >= mark)
			{
				h_samplelights[g*sunray.num_sunshape_lights_per_group_ + size] = global_func::angle2xyz(make_float2(h_tmp_theta[i], h_tmp_phi[i]));
				++size;
			}
		}
	}

	// Step 3 : Transfer h_samplelights to sunray.d_samplelights_
	global_func::cpu2gpu(sunray.d_samplelights_, h_samplelights, num_all_lights);

	//	Step 4 : Cleanup
	delete[] h_samplelights;
	delete[] h_tmp_theta;
	delete[] h_tmp_phi;
	delete[] Intens;

	h_samplelights = nullptr;
	h_tmp_theta = nullptr;
	h_tmp_phi = nullptr;
	Intens = nullptr;
}

//void SceneProcessor::set_samplelights(SunRay &sunray)
//{
//	int num_all_lights = sunray.num_sunshape_groups_ * sunray.num_sunshape_lights_per_group_;
//	float3 *h_angle = new float3[num_all_lights];
//	for (int i = 0; i < num_all_lights; ++i)
//	{
//		h_angle[i].x = 0;
//		h_angle[i].y = 1;
//		h_angle[i].z = 0;
//	}
//	global_func::cpu2gpu(sunray.d_samplelights_, h_angle, num_all_lights);
//
//	delete[] h_angle;
//	h_angle = nullptr;
//}