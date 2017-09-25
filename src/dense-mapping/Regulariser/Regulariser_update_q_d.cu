#include <opencv2/gpu/device/common.hpp> // for cudaSafeCall
#include <opencv2/core/core.hpp> // for CV_Assert
#include "Regulariser.cuh"

static __global__ void update_q(float *g, float *a,  // const input
                                float *q, float *d,  // input  q, d
                                int w, int h, // dimensions: width, height
                                float sigma_q, float sigma_d, float epsilon, float theta // parameters
                                )
{
	// thread coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int i  = (y * w + x);
	const int wh = (w*h);

	// gradients dd_x := $\partial_{x}^{+}d$ computed using forward differences
	float dd_x = (x==w-1)? 0.0f : d[i+1] - d[i];
	float dd_y = (y==h-1)? 0.0f : d[i+w] - d[i];

	float qx = (q[i]    + sigma_q*g[i]*dd_x) / (1.0f + sigma_q*epsilon);
	float qy = (q[i+wh] + sigma_q*g[i]*dd_y) / (1.0f + sigma_q*epsilon);

	// q reprojected **element-wise** as per Newcombe thesis pg. 76, 79 (sec. 3.5)
	// if the whole vector q had to be reprojected, a tree-reduction sum would have been required
	float maxq = fmaxf(1.0f, sqrtf(qx*qx + qy*qy));
	q[i]    = qx / maxq;
	q[i+wh] = qy / maxq;
}

static __global__ void update_d(float *g, float *a,  // const input
                                float *q, float *d,  // input  q, d
                                int w, int h, // dimensions: width, height
                                float sigma_q, float sigma_d, float epsilon, float theta // parameters
                                )
{
	// thread coordinates
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int i  = (y * w + x);
	const int wh = (w*h);

	// div_q computed using backward differences
	float dqx_x = (x==0)? q[i]    - q[i+1]    : q[i]    - q[i-1];
	float dqy_y = (y==0)? q[i+wh] - q[i+wh+w] : q[i+wh] - q[i+wh-w];
	float div_q = dqx_x + dqy_y;

	d[i]  = (d[i] + sigma_d*(g[i]*div_q + a[i]/theta)) / (1.0f + sigma_d/theta);
}

void update_q_dCaller(float *g, float *a,  // const input
                      float *q,  float *d,  // input q, d
                      int width, int height, // dimensions
                      float sigma_q, float sigma_d, float epsilon, float theta // parameters
                      )
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
				 (height + dimBlock.y - 1) / dimBlock.y);

	update_q<<<dimGrid, dimBlock>>>(g, a,  // const input
									q, d,  // input  q, d
									width, height, // dimensions: width, height
									sigma_q, sigma_d, epsilon, theta // parameters
									);
	cudaDeviceSynchronize();
	cudaSafeCall( cudaGetLastError() );

	update_d<<<dimGrid, dimBlock>>>(g, a,  // const input
									q, d,  // input  q, d
									width, height, // dimensions: width, height
									sigma_q, sigma_d, epsilon, theta // parameters
									);
	cudaDeviceSynchronize();
	cudaSafeCall( cudaGetLastError() );
}

