#include <opencv2/gpu/device/common.hpp> //for cudaSafeCall
#include <opencv2/core/core.hpp> //for CV_Assert
#include "update_qd.cuh"

// TODO make sure that gpu memory is not leaked by either reusing memory or freeing it

// TODO check that allocated gpu memory is continuous

// TODO both the width of the thread block and the width of the array must be a multiple of the warp size. round up width to the closest multiple of warp size and pad rows accordingly

// TODO check that output of update_qdNewCaller and update_qdCaller are the same

static __global__ void update_qd(float *g, float *a,  // const input
                                 float *q, float *d,  // input  q, d
                                 int w, int h, // dimensions: width, height
                                 float sigma_q, float sigma_d, float epsilon, float theta // parameters
                                 )
{
  // Calculate texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = (y * w + x);
  const int wh = (w*h);

  // gradients dd_x := $\partial_{x}^{+}d$ computed using forward differences
  float dd_x = (x==w-1)? 0.0f : d[i+1] - d[i];
  float dd_y = (y==h-1)? 0.0f : d[i+w] - d[i];

  float qx = (q[i]    + sigma_q*g[i]*dd_x) / (1.0f + sigma_q*epsilon);
  float qy = (q[i+wh] + sigma_q*g[i]*dd_y) / (1.0f + sigma_q*epsilon);

  // reproject q **element-wise**
  // if the whole vector q had to be reprojected, a tree-reduction sum would have been required
  float maxq = fmaxf(1.0f, sqrtf(qx*qx + qy*qy));
  q[i]    = qx / maxq;
  q[i+wh] = qy / maxq;

  __syncthreads();

  // div_q1 computed using backward differences
  float dq1x_x = (x==0)? q[i]    : q[i]    - q[i-1];
  float dq1y_y = (y==0)? q[i+wh] : q[i+wh] - q[i+wh-w];
  float div_q1 = dq1x_x + dq1y_y;

  d[i]  = (d[i] + sigma_d*(g[i]*div_q1 + a[i]/theta)) / (1.0f + sigma_d/theta);

  __syncthreads();
}

void update_qdCaller(float *g, float *a,  // const input
                     float *q,  float *d,  // input q, d
                     int width, int height, // dimensions
                     float sigma_q, float sigma_d, float epsilon, float theta // parameters
                     )
{
  dim3 dimBlock(32, 32);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);

  update_qd<<<dimGrid, dimBlock>>>(g, a,  // const input
                                   q, d,  // input  q, d
                                   width, height, // dimensions: width, height
                                   sigma_q, sigma_d, epsilon, theta // parameters
                                   );

  cudaDeviceSynchronize();
  cudaSafeCall( cudaGetLastError() );
}

static __global__ void update_qNoTex(float *g, float *a,  // const input
                                 float *q, float *d,  // input  q, d
                                 int w, int h, // dimensions: width, height
                                 float sigma_q, float sigma_d, float epsilon, float theta // parameters
                                 )
{
  // Calculate texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = (y * w + x);
  const int wh = (w*h);

  // gradients dd_x := $\partial_{x}^{+}d$ computed using forward differences
  float dd_x = (x==w-1)? 0.0f : d[i+1] - d[i];
  float dd_y = (y==h-1)? 0.0f : d[i+w] - d[i];

  float qx = (q[i]    + sigma_q*g[i]*dd_x) / (1.0f + sigma_q*epsilon);
  float qy = (q[i+wh] + sigma_q*g[i]*dd_y) / (1.0f + sigma_q*epsilon);

  // reproject q **element-wise**
  // if the whole vector q had to be reprojected, a tree-reduction sum would have been required
  float maxq = fmaxf(1.0f, sqrtf(qx*qx + qy*qy));
  q[i]    = qx / maxq;
  q[i+wh] = qy / maxq;
}

static __global__ void update_dNoTex(float *g, float *a,  // const input
                                 float *q, float *d,  // input  q, d
                                 int w, int h, // dimensions: width, height
                                 float sigma_q, float sigma_d, float epsilon, float theta // parameters
                                 )
{
  // Calculate texture coordinates
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

void update_q_d_NoTexCaller(float *g, float *a,  // const input
                            float *q,  float *d,  // input q, d
                            int width, int height, // dimensions
                            float sigma_q, float sigma_d, float epsilon, float theta // parameters
                            )
{
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);

  update_qNoTex<<<dimGrid, dimBlock>>>(g, a,  // const input
                                   q, d,  // input  q, d
                                   width, height, // dimensions: width, height
                                   sigma_q, sigma_d, epsilon, theta // parameters
                                   );

  cudaDeviceSynchronize();
  cudaSafeCall( cudaGetLastError() );

  update_dNoTex<<<dimGrid, dimBlock>>>(g, a,  // const input
                                   q, d,  // input  q, d
                                   width, height, // dimensions: width, height
                                   sigma_q, sigma_d, epsilon, theta // parameters
                                   );
  cudaDeviceSynchronize();
  cudaSafeCall( cudaGetLastError() );
}

void update_qdCPU(float *g, float *a,  // const input
                  float *q,  float *d,  // input q, d
                  int w, int h, // dimensions
                  float sigma_q, float sigma_d, float epsilon, float theta // parameters
                  )
{
  const int wh = w*h;

  int i;
  for(int x=0; x<w; ++x) {
      for(int y=0; y<h; ++y) {
        i = y*w+x;
        
        // gradients dd_x := $\partial_{x}^{+}d$ computed using forward differences
        float dd_x = (x==w-1)? 0.0f : d[i+1] - d[i];
        float dd_y = (y==h-1)? 0.0f : d[i+w] - d[i];
        
        float qx = (q[i]    + sigma_q*g[i]*dd_x) / (1.0f + sigma_q*epsilon);
        float qy = (q[i+wh] + sigma_q*g[i]*dd_y) / (1.0f + sigma_q*epsilon);

        float maxq = max(1.0f, sqrt(qx*qx + qy*qy));
        q[i]    = qx / maxq;
        q[i+wh] = qy / maxq;
      }
  }

  for(int x=0; x<w; ++x) {
      for(int y=0; y<h; ++y) {
        i = y*w+x;
        
        // div_q computed using backward differences
        float dqx_x = (x==0)? q[i]    - q[i+1]    : q[i]    - q[i-1];
        float dqy_y = (y==0)? q[i+wh] - q[i+wh+w] : q[i+wh] - q[i+wh-w];
        float div_q = dqx_x + dqy_y;

        d[i]  = (d[i] + sigma_d*(g[i]*div_q + a[i]/theta)) / (1.0f + sigma_d/theta);
      }
  }
}

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texq; // q is a 2*height x width array, qx top half, qy bottom half
texture<float, cudaTextureType2D, cudaReadModeElementType> texd; 

static __global__ void update_q(float *q, float *g,   // input
                                float width, float height,
                                float sigma_q, float epsilon // parameters
                                )
{
  // Calculate texture coordinates
  const float x = (float) (blockIdx.x * blockDim.x + threadIdx.x);
  const float y = (float) (blockIdx.y * blockDim.y + threadIdx.y);
  
  const int i  = (int)(y * width + x);
  const int hw = (int)(height*width);

  // gradients dd_x := $\partial_{x}^{+}d$ computed using forward differences
  float dd_x = tex2D(texd, x+1, y  ) - tex2D(texd, x, y); // Out of border texture references return 0
  float dd_y = tex2D(texd, x  , y+1) - tex2D(texd, x, y);

  float qx = (q[i]    + sigma_q*g[i]*dd_x) / (1.0f + sigma_q*epsilon);
  float qy = (q[i+hw] + sigma_q*g[i]*dd_y) / (1.0f + sigma_q*epsilon);

  // reproject q **element-wise**
  // if the whole vector q had to be reprojected, a tree-reduction sum would have been required
  float maxq = fmaxf(1.0f, sqrtf(qx*qx + qy*qy));
  q[i]    = qx / maxq;
  q[i+hw] = qy / maxq;
}

static __global__ void update_d(float *d, float *g, float *a,  // input, TODO q1 available from texture memory
                                float width, float height, 
                                float sigma_d, float theta // parameters
                                )
{
  // Calculate texture coordinates
  const float x = (float) (blockIdx.x * blockDim.x + threadIdx.x);
  const float y = (float) (blockIdx.y * blockDim.y + threadIdx.y);

  const int i = (int)(y * width + x);
  
  // div_q1 computed using backward differences
  // texq1 is bound to q1, which is a 2*height x width array
  float dqx_x = tex2D(texq, x, y) - tex2D(texq, x-1, y); // Out of border references return 0
  float dqy_y = (y==0)? (tex2D(texq, x, y+height) - tex2D(texq, x, y+height+1)) :
                        (tex2D(texq, x, y+height) - tex2D(texq, x, y+height-1)) ;

  float div_q = dqx_x + dqy_y;

  d[i]  = (d[i] + sigma_d*(g[i]*div_q + a[i]/theta)) / (1.0f + sigma_d/theta);
}

static cudaChannelFormatDesc _channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
static size_t _offsetd = 0;
static size_t _offsetq = 0;

void update_q_d_BindTextures(float *q,  float *d, int width, int height, int pitch)
{
  // Set texture reference parameters for d
  texd.normalized     = false;
  texd.addressMode[0] = cudaAddressModeClamp; // Out of border texture references return 0
  texd.addressMode[1] = cudaAddressModeClamp;
  texd.filterMode     = cudaFilterModeLinear;

  // Set texture reference parameters for q
  texq.normalized     = false;
  texq.addressMode[0] = cudaAddressModeClamp;
  texq.addressMode[1] = cudaAddressModeClamp;
  texq.filterMode     = cudaFilterModeLinear;
}

// TODO put the parameters and constants in shared (constant?) memory
// TODO move this to a .cuh file
// TODO won't writing q1, d1 to global memory be a bottleneck? are the writes coalesed?
// TODO bind textures only once, and unbind them at program exit, since inplace updates are being used
void update_q_dCaller(float *g,  float *a,  // input
                      float *q,  float *d,  // input  q, d
                      float width, float height, float pitch, // dimensions
                      float sigma_q, float sigma_d, float epsilon, float theta // parameters
                      )
{
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);

  cudaBindTexture2D(&_offsetq, texq, q, _channelDesc, width, 2*height, pitch);
  update_q<<<dimGrid, dimBlock>>>(q, g, width, height, sigma_q, epsilon);
  cudaDeviceSynchronize();
  cudaUnbindTexture(texq);
  cudaSafeCall( cudaGetLastError() );
  
  cudaBindTexture2D(&_offsetd, texd, d, _channelDesc, width, height, pitch);
  update_d<<<dimGrid, dimBlock>>>(d, g, a, width, height, sigma_d, theta);
  cudaDeviceSynchronize();
  cudaUnbindTexture(texd);
  cudaSafeCall( cudaGetLastError() );
}

//---------------------------------- original code ----------------------------
__device__ inline float reproject(float x){
  return x/fmaxf(1.0f,fabsf(x));
}

const int BLOCKX2D=32;
const int BLOCKY2D=32;

static __global__ void updateQ(float* gqxpt, float* gqypt, float *dpt, float * apt,
                               float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                               float theta)
{

//TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch
#if __CUDA_ARCH__>=300
  __shared__ float s[32*BLOCKY2D];
  int x = threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  bool rt   = (x==31);
  bool bbtm = (threadIdx.y==blockDim.y-1);

  int pt, bpt, bdnoff, dnoff;
    
  float tmp;
  bpt = threadIdx.x+threadIdx.y*blockDim.x;
  bdnoff=blockDim.x;
  dnoff=(y<gridDim.y*blockDim.y-1)*cols;

  pt=x+y*cols;

  float dh,dn;
  dn=dpt[pt];

  for(;x<cols;x+=32){
    float qx,gx,gqx,qy,gy,gqy;
    pt=x+y*cols;

    { //qx update
      float dr;
      { //load(dh,dn,gxh,gqx);//load here, next(the block to the right), local constant, old x force(with cached multiply)
        dh=dn;
        if(x<cols-32){
          dn=dpt[pt+32];
        }
        gqx=gqxpt[pt];
        gx=gxpt[pt]+.01f;
      }

      dr  = __shfl_down(dh,1);
      tmp = __shfl_up(dn,31);
      if (rt && x<cols-32)
        dr=tmp;
      qx = gqx/gx;
      qx = (qx+sigma_q*gx*(dr-dh))/(1+sigma_q*epsilon); //basic spring force equation f=k(x-x0)
      gqx = reproject(gx*qx); //spring reprojects (with cached multiply), saturation force proportional to prob. of not an edge.
      gqxpt[pt]=gqx;
    }
    
    { // qy update
      float dd;
      { //load
        gqy=gqypt[pt];
        gy=gypt[pt]+.01f;
      }
      s[bpt]=dh;
      __syncthreads();
      if(!bbtm)
        dd=s[bpt+bdnoff];
      else
        dd=dpt[pt+dnoff];
      __syncthreads();
      qy = gqy/gy;
      qy = (qy+sigma_q*gy*(dd-dh))/(1+sigma_q*epsilon);
      gqy = reproject(gy*qy);
      gqypt[pt]=gqy;
    }
  }
#endif
}

static __global__ void updateD(float* gqxpt, float* gqypt, float *dpt, float * apt,
                               float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                               float theta)
{
#if __CUDA_ARCH__>=300
  //TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch

  __shared__ float s[32*BLOCKY2D];
  int x = threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  bool lf=x==0;
  bool top=y==0;
  bool btop=threadIdx.y==0;
  int pt, bpt, bupoff, upoff;


  float gqsave=0;
  bpt = threadIdx.x+threadIdx.y*blockDim.x;

  bupoff=-blockDim.x;
  upoff=-(!top)*cols;

  pt=x+y*cols;

  for(;x<cols;x+=32) {
    float gqx,gqy;
    pt=x+y*cols;


    float dacc;
    //dx update
    {
      float gqr,gql;
      gqr=gqx=gqxpt[pt];
      gql=__shfl_up(gqx,1);
      if (lf)
        gql=gqsave;
      gqsave=__shfl_down(gqx,31);//save for next iter
      dacc = gqr - gql;//dx part
    }
    //dy update and d store
    { //load
      float a=apt[pt];
      float gqu,gqd;
      float d=dpt[pt];
      gqd=gqy=gqypt[pt];
      s[bpt]=gqy;
      __syncthreads();
      if(!btop)
        gqu=s[bpt+bupoff];
      else
        gqu=gqypt[pt + upoff];
      if(y==0)
        gqu=0.0f;
      dacc += gqd-gqu; //dy part
      //d += dacc*.5f;//simplified step
      d = ( d + sigma_d*(dacc + a/theta) ) / (1 + sigma_d/theta);

      dpt[pt] = d;
    }
    __syncthreads();//can't figure out why this is needed, but it is to avoid subtle errors in Qy at the ends of the warp
  }
#endif
}

// TODO: make sure that rows is passed at call sites, and also change signature in updateQD.cuh
void updateQDCaller(float* gqxpt, float* gqypt, float *dpt, float * apt,
                    float *gxpt, float *gypt, int cols, int rows, float sigma_q, float sigma_d, float epsilon, float theta)
{
  dim3 dimBlock(BLOCKX2D, BLOCKY2D);
  dim3 dimGrid(1, (rows + dimBlock.y - 1) / dimBlock.y);
  CV_Assert(dimGrid.y>0);
  cudaSafeCall( cudaGetLastError() );
  updateQ<<<dimGrid, dimBlock>>>(gqxpt, gqypt, dpt, apt,
                                 gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta);
  cudaSafeCall( cudaGetLastError() );
  updateD<<<dimGrid, dimBlock>>>(gqxpt, gqypt, dpt, apt,
                                 gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta);
  cudaDeviceSynchronize();
  cudaSafeCall( cudaGetLastError() );
}
