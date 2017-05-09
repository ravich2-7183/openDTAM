#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "CostVolume.cuh"

#define SET_Z_START()								\
	float d_start = di - fabsf(di-dmin);			\
	z = lrintf(floorf((d_start - far)/depthStep));	\
	z = (z<0)? 0 : z;								\
	z_start = z;									\

#define SET_Z_END()								\
	float d_end = di + fabsf(di-dmin);			\
	z = lrintf(ceilf((d_end - far)/depthStep));	\
	z = (z>(layers-1))? (layers-1) : z;			\
	z_end = z;									\

__device__
static inline float Eaux(float theta, float di, float aIdx, float far, float depthStep, float lambda, float costval)
{
    float ai = far + float(aIdx)*depthStep;
    return (0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval; // TODO beware float substraction
}

static __global__ void minimizeA(float* cdata, int rows, int cols,
                                 float* a, float* d,
								 float* d_Cmin,
                                 float far, float depthStep, int layers,
                                 float theta, float lambda)
{
    // thread coordinate
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = x + y*cols;

    const int   layerStep = rows*cols;
    const float di        = d[i];
    const float dmin      = d_Cmin[i];

    int   minz = 0;
    float minv = 1e+30;
    // #pragma unroll 4 // TODO what does the 4 do?
	int z, z_start, z_end;
	SET_Z_START();
	SET_Z_END();
    for(int z=z_start; z<=z_end; z++) {
        float c = Eaux(theta, di, z, far, depthStep, lambda, cdata[i+z*layerStep]);
        if(c < minv) {
            minv = c;
            minz = z;
        }
    }
	
	a[i] = far + float(minz)*depthStep;
	
    if(minz == 0 || minz == layers-1) // first or last was best
        return;
    
    // sublayer sampling as the minimum of the parabola with the 2 points around (minz, minv)
    float A = Eaux(theta, di, minz-1, far, depthStep, lambda, cdata[i+(minz-1)*layerStep]);
    float B = minv;
    float C = Eaux(theta, di, minz+1, far, depthStep, lambda, cdata[i+(minz+1)*layerStep]);
    float delta = ((A+C)==2*B)? 0.0f : ((A-C)*depthStep)/(2*(A-2*B+C));
    a[i] += delta;
}

void minimizeACaller(float*cdata, int rows, int cols, int layers, 
                     float*a, float*d,
					 float*d_Cmin,
                     float far, float depthStep,
                     float theta, float lambda)
{ 
   dim3 dimBlock(16, 16);
   dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                (rows + dimBlock.y - 1) / dimBlock.y);

  minimizeA<<<dimGrid, dimBlock>>>(cdata, rows, cols, 
                                   a, d, 
								   d_Cmin,
								   far, depthStep, layers,
								   theta, lambda);
  
  cudaDeviceSynchronize();
  cudaSafeCall( cudaGetLastError() ); 
}
