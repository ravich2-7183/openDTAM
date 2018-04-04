#ifndef TESTTEXTURE_CUH
#define TESTTEXTURE_CUH

#include <opencv2/gpu/device/common.hpp>

void testTextureCaller(int rows, int cols, int imageStep, float4* current_image, float4* output_image);

#endif
