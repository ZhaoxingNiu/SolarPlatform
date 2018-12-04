#include "./reduce_sum.cuh"

const int threadsPerBlockReduceSum = 1024;
// 可以进行优化的
__global__ void ReductionSum(float *d_a, float *d_partial_sum)
{
	//申请共享内存，存在于每个block中
	__shared__ float partialSum[threadsPerBlockReduceSum];

	//确定索引
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	//传global memory数据到shared memory
	partialSum[tid] = d_a[i];

	//传输同步
	__syncthreads();

	//在共享存储器中进行规约
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
	if (tid < stride) partialSum[tid] += partialSum[tid + stride];
	__syncthreads();
	}
	//将当前block的计算结果写回输出数组
	if (tid == 0)
	d_partial_sum[blockIdx.x] = partialSum[0];
}



float get_discrete_area(float *d_data, int N, float pixel_length) {
	// decide the grid's number
	const int blocksPerGrid = (N + threadsPerBlockReduceSum - 1) / threadsPerBlockReduceSum;

	// 申请host 端内存和初始化
	float  *h_partial_sum, *d_partial_sum;
	h_partial_sum = (float*)malloc(blocksPerGrid * sizeof(float));
	cudaMalloc((void**)&d_partial_sum, blocksPerGrid * sizeof(float));

	//调用内核函数
	ReductionSum<<<blocksPerGrid, threadsPerBlockReduceSum>>>(d_data, d_partial_sum);

	//将结果传回到主机端
	cudaMemcpy(h_partial_sum, d_partial_sum, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);

	//将部分和求和
	float sum = 0;
	for (int i = 0; i < blocksPerGrid; ++i) {
		sum += h_partial_sum[i];
	}

	float area = sum * pixel_length * pixel_length;
	return area;
}
