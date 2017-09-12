#include <opencv2/core/core.hpp>
#include "CostVolume.cuh"

// 2D float texture
static texture<float4, cudaTextureType2D, cudaReadModeElementType> currentImageTexRef;

static inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,	 a.w + b.w);
}

static inline __host__ __device__ float4 operator*(float4 a, float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

static inline __host__ __device__ float4 operator*(float b, float4 a)
{
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

// In case texture memory improperly behaves
// Boundary checking is done in the calling function
#define BILINEAR_INTERPOLATE(A, x, y)                   \
    // float xmax = float(cols);                        \
    // float ymax = float(rows);                        \
    //                                                  \
    // float xf, yf;                                    \
    // xf = (x < 0.0f) ? 0.0f : x;                      \
    // xf = (x > xmax) ? xmax : x;                      \
    // yf = (y < 0.0f) ? 0.0f : y;                      \
    // yf = (y > ymax) ? ymax : y;                      \
    //                                                  \
    float x1f = floorf(xf); int x1 = lrintf(x1f);       \
    float x2f =  ceilf(xf); int x2 = lrintf(x2f);       \
    float y1f = floorf(yf); int y1 = lrintf(y1f);       \
    float y2f =  ceilf(yf); int y2 = lrintf(y2f);       \
                                                        \
    float4 f11 = A[x1 + y1*cols];                       \
    float4 f12 = A[x1 + y2*cols];                       \
    float4 f21 = A[x2 + y1*cols];                       \
    float4 f22 = A[x2 + y2*cols];                       \
                                                        \
    if(x1 == x2 && y1 == y2)                            \
        Im = f11;                                       \
    if(x1 == x2 && y1 != y2)                            \
        Im = f11*(y2f-yf) + f12*(yf-y1f);               \
    if(x1 != x2 && y1 == y2)                            \
        Im = f11*(x2f-xf) + f21*(xf-x1f);               \
    if(x1 != x2 && y1 != y2)                            \
        Im = (f11*((x2f-xf)*(y2f-yf)) +                 \
              f21*((xf-x1f)*(y2f-yf)) +                 \
              f12*((x2f-xf)*(yf-y1f)) +                 \
              f22*((xf-x1f)*(yf-y1f)));                 \

// TODO: replace the floats with floats and check for any performance increase vs. lost accuracy 
static __global__ void updateCostVolume(float* K, float* Kinv, float* Tmr,
										int rows, int cols,
										float near, float far, int layers, int layerStep,
                                        float* Cost, float count,
										float* Cmin, float* Cmax, float* CminIdx,
										float4* referenceImage, float4* currentImage, bool useTextureMemory)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x + y*cols;

	// TODO test the use of float vs float: precision vs. speed. My guess is that float should be enough. 
	const float ur = x;
	const float vr = y;

	const float depthStep = (near - far)/(layers-1);

	float4 Ir = referenceImage[i];

    int	  minl = layers-1; // TODO set to layers?
    float Cost_min = 1e+30, Cost_max = 0.0;
    for(int l=layers-1; l >= 0; l--) { // TODO march from front to back, i.e., l = layers -> 0 and check results. 
        float d = far + float(l)*depthStep;
		// 0 1 2
		// 3 4 5
		// 6 7 8
		float zr = 1.0/d; // divide by 0 is evaluated as Inf, as per IEEE-754
		float xr = (Kinv[0]*ur + Kinv[2])*zr;
		float yr = (Kinv[4]*vr + Kinv[5])*zr;
		//  0  1  2  3
		//  4  5  6  7
		//  8  9 10 11
		// 12 13 14 15
		float xm = Tmr[0]*xr + Tmr[1]*yr + Tmr[2]*zr  + Tmr[3];
		float ym = Tmr[4]*xr + Tmr[5]*yr + Tmr[6]*zr  + Tmr[7];
		float zm = Tmr[8]*xr + Tmr[9]*yr + Tmr[10]*zr + Tmr[11];
		// 0 1 2
		// 3 4 5
		// 6 7 8
		float um = K[0]*(xm/zm) + K[2];
		float vm = K[4]*(ym/zm) + K[5];

		// TODO uncomment these lines and check results
		if( (um > float(cols)) || (um < 0.0f) || (vm > float(rows)) || (vm < 0.0f) )
			continue;

		// TODO if corresponding pixel is out of the image, then apply a penalty. 
		// TODO right now this is being rewarded, as nothing is added to the cost in this case.
		// TODO best way might be to use a per pixel count, recording number of hits. requires more careful thought. 

		// TODO use each exclusively and compare results
		float4 Im;
		if(useTextureMemory) {
			Im = tex2D(currentImageTexRef, um, vm);
		}
		else {
			BILINEAR_INTERPOLATE(currentImage, um, vm);
		}

		float rho = fabsf(Im.x - Ir.x) + fabsf(Im.y - Ir.y) + fabsf(Im.z - Ir.z);
        Cost[i+l*layerStep] = (Cost[i+l*layerStep]*(count-1) + rho) / count; // TODO: maintain per pixel count?
        float Cost_l = Cost[i+l*layerStep];
        if(Cost_l <= Cost_min) {
            Cost_min = Cost_l;
            minl = l;
		}
        Cost_max = fmaxf(Cost_l, Cost_max);
	}

    Cmin[i]	   = Cost_min;
    CminIdx[i] = far + float(minl)*depthStep; // scaling is done when used in DepthEstimator::optimize
    Cmax[i]	   = Cost_max;

	// sublayer sampling as the minimum of the parabola with the 2 points around (minz, minv)
	if(minl == 0 || minl == layers-1) // first or last was best
		return;

    float A = far + float(minl-1)*depthStep;
    float B = CminIdx[i];
    float C = far + float(minl+1)*depthStep;
	float delta = ((A+C)==2*B)? 0.0f : ((A-C)*depthStep)/(2*(A-2*B+C));
	delta = (fabsf(delta) > depthStep)? 0.0f : delta;
	CminIdx[i] += delta;
}

void updateCostVolumeCaller(float* K, float* Kinv, float* Tmr,
							int rows, int cols, int imageStep,
							float near, float far, int layers, int layerStep,
							float* Cdata, float count,
							float* Cmin, float* Cmax, float* CminIdx,
							float4* referenceImage, float4* currentImage, bool useTextureMemory)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
				 (rows + dimBlock.y - 1) / dimBlock.y);

	if(useTextureMemory) {
		// Set texture reference parameters
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

		currentImageTexRef.normalized	  = false;
		currentImageTexRef.addressMode[0] = cudaAddressModeClamp; // out of border references return first or last element
		currentImageTexRef.addressMode[1] = cudaAddressModeClamp;
		currentImageTexRef.filterMode	  = cudaFilterModeLinear;

		// Bind currentImage to the texture reference
		size_t offset;
		cudaBindTexture2D(&offset, currentImageTexRef, currentImage, channelDesc, cols, rows, imageStep);

		cudaDeviceSynchronize();
		cudaSafeCall(cudaGetLastError());
	}

	updateCostVolume<<<dimGrid, dimBlock>>>(K, Kinv, Tmr,
											rows, cols,
											near, far, layers, layerStep,
											Cdata, count,
											Cmin, Cmax, CminIdx,
											referenceImage, currentImage, useTextureMemory);
	cudaDeviceSynchronize();
	cudaSafeCall(cudaGetLastError());
	if(useTextureMemory)
		cudaUnbindTexture(currentImageTexRef);
}
