#include "./reduce_test.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

const static int threadsPerBlock = 512;
const static int N = 2048;
const static int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

__global__ void ReductionSum1(float *d_a, float *d_partial_sum)
{
	//���빲���ڴ棬������ÿ��block�� 
	__shared__ float partialSum[threadsPerBlock];

	//ȷ������
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	//��global memory���ݵ�shared memory
	partialSum[tid] = d_a[i];

	//����ͬ��
	__syncthreads();

	//�ڹ���洢���н��й�Լ
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride) partialSum[tid] += partialSum[tid + stride];
		__syncthreads();
	}
	//����ǰblock�ļ�����д���������
	if (tid == 0)
		d_partial_sum[blockIdx.x] = partialSum[0];
}

int testGpuReduce()
{
	//����host���ڴ漰��ʼ��
	float   *h_a, *h_partial_sum;
	h_a = (float*)malloc(N * sizeof(float));
	h_partial_sum = (float*)malloc(blocksPerGrid * sizeof(float));

	for (int i = 0; i < N; ++i)  h_a[i] = 1;

	//�����Դ�ռ�
	int size = sizeof(float);
	float *d_a;
	float *d_partial_sum;
	cudaMalloc((void**)&d_a, N*size);
	cudaMalloc((void**)&d_partial_sum, blocksPerGrid*size);

	//�����ݴ�Host����Device
	cudaMemcpy(d_a, h_a, size*N, cudaMemcpyHostToDevice);

	//�����ں˺���
	ReductionSum1 <<<blocksPerGrid, threadsPerBlock >>> (d_a, d_partial_sum);

	//��������ص�������
	cudaMemcpy(h_partial_sum, d_partial_sum, size*blocksPerGrid, cudaMemcpyDeviceToHost);

	//�����ֺ����
	int sum = 0;
	for (int i = 0; i < blocksPerGrid; ++i)  sum += h_partial_sum[i];

	cout << "sum=" << sum << endl;

	//�ͷ��Դ�ռ�
	cudaFree(d_a);
	cudaFree(d_partial_sum);

	free(h_a);
	free(h_partial_sum);

	return 0;
}

