#include "./reduce_sum.cuh"

const int threadsPerBlockReduceSum = 1024;
// ���Խ����Ż���
__global__ void ReductionSum(float *d_a, float *d_partial_sum)
{
	//���빲���ڴ棬������ÿ��block��
	__shared__ float partialSum[threadsPerBlockReduceSum];

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



float get_discrete_area(float *d_data, int N, float pixel_length) {
	// decide the grid's number
	const int blocksPerGrid = (N + threadsPerBlockReduceSum - 1) / threadsPerBlockReduceSum;

	// ����host ���ڴ�ͳ�ʼ��
	float  *h_partial_sum, *d_partial_sum;
	h_partial_sum = (float*)malloc(blocksPerGrid * sizeof(float));
	cudaMalloc((void**)&d_partial_sum, blocksPerGrid * sizeof(float));

	//�����ں˺���
	ReductionSum<<<blocksPerGrid, threadsPerBlockReduceSum>>>(d_data, d_partial_sum);

	//��������ص�������
	cudaMemcpy(h_partial_sum, d_partial_sum, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);

	//�����ֺ����
	float sum = 0;
	for (int i = 0; i < blocksPerGrid; ++i) {
		sum += h_partial_sum[i];
	}

	float area = sum * pixel_length * pixel_length;
	return area;
}
