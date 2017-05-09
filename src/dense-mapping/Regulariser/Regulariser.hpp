#ifndef Regulariser_H
#define Regulariser_H

#include <opencv2/gpu/gpu.hpp>

using namespace cv::gpu;

class Regulariser
{
public:
	Regulariser() {};
	Regulariser(float rows, float cols, float alphaG, float betaG);
	
	void initialize(const GpuMat& referenceImageGray);
	void update_q_d(const GpuMat& a, GpuMat& d, float epsilon, float theta);

	GpuMat q_, g_;
	float epsilon_;
	float alphaG_, betaG_;
	float sigma_d_, sigma_q_;

private:
	void computeSigmas(float epsilon, float theta);
}; 
#endif // Regulariser_H
