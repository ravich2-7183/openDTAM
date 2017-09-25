#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH

#include <opencv2/gpu/device/common.hpp>

void updateCostVolumeCaller(float* K, float* Kinv, float* Tmr,
                            int rows, int cols, int imageStep, 
                            float near, float far, int layers, int layerStep,
                            float* Cdata, float count, 
							float* Cmin, float* Cmax, float* CminIdx,
                            float* reference_image_gray, float* current_image_gray);

void minimizeACaller(float*cdata, int rows, int cols,
                     float*a, float*d,
					 float*d_Cmin, float*C_min, float*C_max,
					 float far, float near, int layers,
                     float theta, float lambda);
#endif
