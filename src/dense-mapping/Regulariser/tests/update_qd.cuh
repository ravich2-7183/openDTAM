
#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH
#include <opencv2/gpu/device/common.hpp>//for cudaStream_t

using namespace std;

// original OpenDTAM
void updateQDCaller(float* gqxpt, float* gqypt, float *dpt, float * apt,
                    float *gxpt, float *gypt, int cols, int rows, float sigma_q, float sigma_d, float epsilon, float theta);

// q and d updates in one kernel
void update_qdCaller(float *g, float *a,  // const input
                     float *q,  float *d,  // input q, d
                     int width, int height, // dimensions
                     float sigma_q, float sigma_d, float epsilon, float theta // parameters
                     );

// without using texture memory
void update_q_d_NoTexCaller(float *g, float *a,  // const input
                            float *q, float *d,  // input q, d
                            int width, int height, // dimensions
                            float sigma_q, float sigma_d, float epsilon, float theta // parameters
                            );

void update_q_d_BindTextures(float *q,  float *d, int width, int height, int pitch);

// using texture memory
void update_q_dCaller(float *g,  float *a,  // input
                      float *q,  float *d,  // input  q, d
                      float width, float height, float pitch, // dimensions
                      float sigma_q, float sigma_d, float epsilon, float theta // parameters
                      );

void update_qdCPU(float *g, float *a,  // const input
                  float *q,  float *d,  // input q, d
                  int w, int h, // dimensions
                  float sigma_q, float sigma_d, float epsilon, float theta // parameters
                  );

#endif
