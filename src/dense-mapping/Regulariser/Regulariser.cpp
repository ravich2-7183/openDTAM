#include "Regulariser.hpp"
#include "Regulariser.cuh"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

Regulariser::Regulariser(float rows, float cols, float alphaG, float betaG) :
	alphaG_(alphaG),
	betaG_(betaG)
{
	// allocate g_ and q_
	cv::gpu::createContinuous(  rows, cols, CV_32FC1, g_);
	cv::gpu::createContinuous(2*rows, cols, CV_32FC1, q_);

	CV_Assert(g_.step == cols*4 && q_.step == cols*4);
}

void Regulariser::initialize(const GpuMat& referenceImageGray)
{
	q_ = 0.0f;
	
	// Call the gpu function for caching g's
	computeGCaller( (float*)referenceImageGray.data,
					(float*)g_.data,
					referenceImageGray.cols, referenceImageGray.rows, referenceImageGray.step, 
					alphaG_, betaG_);
}

void Regulariser::computeSigmas(float epsilon, float theta)
{
    /*
	  The DTAM paper only provides a reference [3] for setting sigma_q & sigma_d
	  
	  [3] A. Chambolle and T. Pock. A first-order primal-dual 
	  algorithm for convex problems with applications to imaging.
	  Journal of Mathematical Imaging and Vision, 40(1):120-
	  145, 2011.
	  
	  The relevant section of this dense paper is:
	  Sec. 6.2.3 The Huber-ROF Model, ALG3
    */

	//lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44        
	float L = 4.0;

	float mu = 2.0*std::sqrt(epsilon/theta)/L;

	// TODO: check the original paper for correctness of these settings
	sigma_d_ = mu/(2.0/theta);
	sigma_q_ = mu/(2.0*epsilon);
}

void Regulariser::update_q_d(const GpuMat& a, GpuMat& d, float epsilon, float theta)
{
	computeSigmas(epsilon, theta);

	// TODO test code
	// cout << "sigma_q = " << sigma_q_ <<  ", sigma_d = " << sigma_d_ << endl;

	update_q_dCaller((float*)g_.data, (float*)a.data,
					 (float*)q_.data, (float*)d.data,
					 a.cols, a.rows,
					 sigma_q_, sigma_d_, epsilon, theta);
}
