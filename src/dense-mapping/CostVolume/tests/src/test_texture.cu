#include <opencv2/core/core.hpp>
#include "testTexture.cuh"

// 2D float texture
static texture<float4, cudaTextureType2D, cudaReadModeElementType> current_imageTexRef;

static __global__ void testTexture(int rows, int cols, float4* output_image)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + y*cols;

	const float u = x;
	const float v = y;

	output_image[i] = tex2D(current_imageTexRef, u + 1, v + 1);
}

void testTextureCaller(int rows, int cols, int imageStep, float4* current_image, float4* output_image)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
				 (rows + dimBlock.y - 1) / dimBlock.y);

	// Set texture reference parameters
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	current_imageTexRef.normalized     = false;
	current_imageTexRef.addressMode[0] = cudaAddressModeClamp;	// out of border references return first or last element
	current_imageTexRef.addressMode[1] = cudaAddressModeClamp;
	current_imageTexRef.filterMode     = cudaFilterModeLinear;

	// Bind current_image to the texture reference
	// size_t offset;
	cudaBindTexture2D(NULL, current_imageTexRef, current_image, channelDesc, cols, rows, imageStep);

	cudaDeviceSynchronize();
	cudaSafeCall(cudaGetLastError());

	testTexture<<<dimGrid, dimBlock>>>(rows, cols, output_image);
	cudaDeviceSynchronize();
	cudaSafeCall(cudaGetLastError());
	cudaUnbindTexture(current_imageTexRef);
}
