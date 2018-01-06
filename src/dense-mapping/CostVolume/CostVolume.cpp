#include "CostVolume.hpp"
#include "CostVolume.cuh"

#include <opencv2/opencv.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/gpu/device/common.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

CostVolume::CostVolume(float _rows, float _cols, float _layers, float _near, float _far, const cv::Mat& Kcpu):
	rows(_rows), cols(_cols),
	layers(_layers), near(_near), far(_far)
{
	CV_Assert(layers >= 8);
	CV_Assert(near > far);

	depthStep = (near - far) / (layers - 1);

	cv::gpu::createContinuous(rows, cols, CV_32FC1, Cmin);
	cv::gpu::createContinuous(rows, cols, CV_32FC1, Cmax);
	cv::gpu::createContinuous(rows, cols, CV_32FC1, CminIdx);

	// TODO reorg as (rows*cols, layers) instead of current (layers, rows*cols)
	cv::gpu::createContinuous(layers, rows*cols, CV_32FC1, cost_data);

	cdata = (float*) cost_data.data;

	cv::gpu::createContinuous(3, 3, CV_32FC1, K);
	cv::gpu::createContinuous(3, 3, CV_32FC1, Kinv);

	cv::gpu::createContinuous(4, 4, CV_32FC1, Tmr_gpu);

	cv::gpu::createContinuous(rows, cols, CV_32FC4, reference_image_color_);
	cv::gpu::createContinuous(rows, cols, CV_32FC1, reference_image_gray_);
	cv::gpu::createContinuous(rows, cols, CV_32FC4, current_image_color_);
	cv::gpu::createContinuous(rows, cols, CV_32FC1, current_image_gray_);

	K.upload(Kcpu);

	float fx, fy, cx, cy;
	fx = Kcpu.at<float>(0,0);
	fy = Kcpu.at<float>(1,1);
	cx = Kcpu.at<float>(0,2);
	cy = Kcpu.at<float>(1,2);

	cv::Mat KInvCpu = (Mat_<float>(3,3) <<
					   1/fx,	0.0, -cx/fx,
					    0.0,   1/fy, -cy/fy,
					    0.0,    0.0,	1.0);

	Kinv.upload(KInvCpu);

	Cmin	= 100;
	CminIdx = near; 
	Cmax	= 200;

	cost_data = 0.0f;
	count_	  = 0;
}

void CostVolume::setReferenceImage(const cv::Mat& reference_image, const cv::Mat& Rrw, const cv::Mat& trw)
{
	CV_Assert(reference_image.type() == CV_32FC4);
	CV_Assert(reference_image.rows % 32 == 0 && reference_image.cols % 32 == 0 && reference_image.cols >= 64);

	Rwr =  Rrw.t();
	twr = -Rrw.t()*trw;
	Twr = (Mat_<float>(4,4) <<
		   Rwr.at<float>(0,0), Rwr.at<float>(0,1), Rwr.at<float>(0,2), twr.at<float>(0),
		   Rwr.at<float>(1,0), Rwr.at<float>(1,1), Rwr.at<float>(1,2), twr.at<float>(1),
		   Rwr.at<float>(2,0), Rwr.at<float>(2,1), Rwr.at<float>(2,2), twr.at<float>(2),
		   0,					0,					 0,				      1);

	reference_image_color_.upload(reference_image);
	cv::gpu::cvtColor(reference_image_color_, reference_image_gray_, CV_RGBA2GRAY); // conversion on gpu presumably faster
}

void CostVolume::reset()
{
	Cmin	= 0.0;
	CminIdx = near; 
	Cmax	= 3.0;

	cost_data = 3;
	count_	  = 0;
}

// TODO to increase throughput, use async functions to copy from host to device
void CostVolume::updateCost(const Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw)
{
	count_++;

	current_image_color_.upload(image);
	cv::gpu::cvtColor(current_image_color_, current_image_gray_, CV_RGBA2GRAY); // conversion on gpu presumably faster

	Tmw = (Mat_<float>(4,4) <<
		   Rmw.at<float>(0,0), Rmw.at<float>(0,1), Rmw.at<float>(0,2), tmw.at<float>(0),
		   Rmw.at<float>(1,0), Rmw.at<float>(1,1), Rmw.at<float>(1,2), tmw.at<float>(1),
		   Rmw.at<float>(2,0), Rmw.at<float>(2,1), Rmw.at<float>(2,2), tmw.at<float>(2),
		   0,                   0,                   0,                1);

	Tmr = Tmw*Twr;
	Tmr_gpu.upload(Tmr);

	// TODO check this carefully, why isn't dataContainer.step == rows*cols?
	updateCostVolumeCaller( (float*)K.data, (float*)Kinv.data, (float*)Tmr_gpu.data,
							rows, cols, current_image_color_.step,
							near, far, layers, rows*cols,
							cdata, count_,
							(float*)(Cmin.data), (float*)(Cmax.data), (float*)(CminIdx.data),
							(float4*)(reference_image_color_.data), (float4*)(current_image_color_.data));
}

void CostVolume::minimize_a(const cv::gpu::GpuMat& d, cv::gpu::GpuMat& a, float theta, float lambda)
{
	minimizeACaller(cdata, rows, cols,
					(float*)a.data, (float*)d.data,
					(float*)CminIdx.data, (float*)Cmin.data, (float*)Cmax.data,
					far, near, layers,
					theta, lambda);
}

CostVolume::~CostVolume()
{}
