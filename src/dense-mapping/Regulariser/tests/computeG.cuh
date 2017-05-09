#ifndef COMPUTEG_CUH
#define COMPUTEG_CUH

#include <opencv2/gpu/device/common.hpp> //for cudaStream_t

void computeGScharrCaller(float* img, float* g, int width, int height, int pitch, float alpha=3.5f, float beta=1.0f);
void computeGCaller(float* pp, float* g1p, float* gxp, float* gyp, int rows, int cols);

#endif
