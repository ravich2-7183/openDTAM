#include <opencv2/gpu/device/common.hpp> // for cudaSafeCall
#include <opencv2/core/core.hpp> // for CV_Assert
#include "Regulariser.cuh"

// TODO pass in alphaG, betaG as parameters
static __global__ void computeG(float* g, float* img, int w, int h, float alphaG, float betaG)
{
  // thread coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = (y * w + x);

  // gradients gx := $\partial_{x}^{+}img$ computed using forward differences
  float gx = (x==w-1)? 0.0f : img[i+1] - img[i];
  float gy = (y==h-1)? 0.0f : img[i+w] - img[i];

  g[i] = expf( -alphaG * powf(sqrtf(gx*gx + gy*gy), betaG) );
}

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Scharr gradient kernel
static __global__ void computeGScharr(float* g, float* img, int w, int h, float alphaG, float betaG)
{
    // Calculate texture coordinates
    float x = (float) (blockIdx.x * blockDim.x + threadIdx.x);
    float y = (float) (blockIdx.y * blockDim.y + threadIdx.y);
    
    const int i = (int)(y * w + x);
    
    // Scharr kernels (combines Gaussian smoothing and differentiation)
    /*  kx           ky
       -3 0  3       -3 -10 -3
      -10 0 10        0   0  0
       -3 0  3        3  10  3
    */

    // Out of border references are clamped to [0, N-1]
    float gx, gy;
    gx = -3.0f * tex2D(texRef, x-1, y-1) +
          3.0f * tex2D(texRef, x+1, y-1) +
         10.0f * tex2D(texRef, x-1, y  ) +
        -10.0f * tex2D(texRef, x+1, y  ) +
         -3.0f * tex2D(texRef, x-1, y+1) +
          3.0f * tex2D(texRef, x+1, y+1) ;

    gy = -3.0f * tex2D(texRef, x-1, y-1) +
         -3.0f * tex2D(texRef, x+1, y-1) +
        -10.0f * tex2D(texRef, x  , y-1) +
         10.0f * tex2D(texRef, x  , y+1) +
          3.0f * tex2D(texRef, x-1, y+1) +
          3.0f * tex2D(texRef, x+1, y+1) ;
    
    g[i] = expf(-alphaG*powf(sqrtf(gx*gx + gy*gy), betaG));
}

void computeGCaller(float* img, float* g,
                    int width, int height, int pitch,
                    float alphaG, float betaG, bool useScharr)
{
  // TODO set dimBlock based on warp size
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);

  if(useScharr) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // Set texture reference parameters
    texRef.normalized     = false;
    texRef.addressMode[0] = cudaAddressModeClamp; // out of border references return first or last element, 
    texRef.addressMode[1] = cudaAddressModeClamp; // this is good enough for Sobel/Scharr filter
    texRef.filterMode     = cudaFilterModeLinear;

    // Bind the array to the texture reference
    size_t offset;
    cudaBindTexture2D(&offset, texRef, img, channelDesc, width, height, pitch);
 
    // Invoke kernel
    computeGScharr<<<dimGrid, dimBlock>>>(g, img, width, height, alphaG, betaG);
    cudaDeviceSynchronize();
    cudaUnbindTexture(texRef);
    cudaSafeCall( cudaGetLastError() );
  }
  else {
    computeG<<<dimGrid, dimBlock>>>(g, img, width, height, alphaG, betaG);
    cudaDeviceSynchronize();
    cudaSafeCall( cudaGetLastError() );
  }
}

