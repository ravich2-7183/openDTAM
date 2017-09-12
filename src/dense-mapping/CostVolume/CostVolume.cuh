#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH

#include <opencv2/gpu/device/common.hpp>

void updateCostVolumeCaller(float* K, float* Kinv, float* Tmr,
                            int rows, int cols, int imageStep, 
                            float near, float far, int layers, int layerStep,
                            float* Cdata, float count, 
							float* Cmin, float* Cmax, float* CminIdx,
                            float4* referenceImage, float4* currentImage, bool useTextureMemory);

void minimizeACaller(float*cdata, int rows, int cols, int layers, 
                     float*a, float*d,
					 float*d_Cmin,
                     float far, float depthStep,
                     float theta, float lambda);
#endif
