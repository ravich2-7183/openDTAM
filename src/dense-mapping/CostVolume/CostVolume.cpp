#include "CostVolume.hpp"
#include "CostVolume.cuh"

#include <opencv2/opencv.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/gpu/device/common.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

void CostVolume::checkInputs(const cv::Mat& R, const cv::Mat& t,
							 const cv::Mat& image,
							 const cv::Mat& cameraMatrix)
{
	CV_Assert(R.size() == Size(3, 3));
	CV_Assert(R.type() == CV_32FC1);
	CV_Assert(t.size() == Size(3, 1) || t.size() == Size(1, 3));
	CV_Assert(t.type() == CV_32FC1);
	CV_Assert(cameraMatrix.size() == Size(3, 3));
	CV_Assert(cameraMatrix.type() == CV_32FC1);

	// TODO remove this requirement
	CV_Assert(image.type() == CV_32FC4);
	CV_Assert(image.rows % 32 == 0 && image.cols % 32 == 0 && image.cols >= 64);
}

CostVolume::CostVolume(float _rows, float _cols, float _layers, float _near, float _far):
	rows(_rows), cols(_cols),
	layers(_layers), near(_near), far(_far)
{
	CV_Assert(layers >= 8);
	
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
	
	cv::gpu::createContinuous(rows, cols, CV_32FC4, referenceImage);
	cv::gpu::createContinuous(rows, cols, CV_32FC4, currentImage);
	cv::gpu::createContinuous(rows, cols, CV_32FC1, referenceImageGray);
}

// TODO: camera doesn't change, so set it only once at startup, instead of at each reset
void CostVolume::reset(const cv::Mat& image, const cv::Mat& Kcpu, const cv::Mat& Rrw, const cv::Mat& trw)
{
	checkInputs(Rrw, trw, image, Kcpu);

	Rwr =  Rrw.t();
	twr = -Rrw.t()*trw;
	Twr = (Mat_<float>(4,4) <<
		   Rwr.at<float>(0,0), Rwr.at<float>(0,1), Rwr.at<float>(0,2), twr.at<float>(0),
		   Rwr.at<float>(1,0), Rwr.at<float>(1,1), Rwr.at<float>(1,2), twr.at<float>(1),
		   Rwr.at<float>(2,0), Rwr.at<float>(2,1), Rwr.at<float>(2,2), twr.at<float>(2),
		   0,					0,					 0,				      1);
	
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

	referenceImage.upload(image);
	cv::gpu::cvtColor(referenceImage, referenceImageGray, CV_RGBA2GRAY); // TODO ensure input image is always RGBA

    cost_data = 0.0;
	
	count = 0.0f;
}

// TODO to increase throughput, use async functions to copy from host to device
void CostVolume::updateCost(const Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw)
{
	count++;

	currentImage.upload(image);

	Tmw = (Mat_<float>(4,4) <<
		   Rmw.at<float>(0,0), Rmw.at<float>(0,1), Rmw.at<float>(0,2), tmw.at<float>(0),
		   Rmw.at<float>(1,0), Rmw.at<float>(1,1), Rmw.at<float>(1,2), tmw.at<float>(1),
		   Rmw.at<float>(2,0), Rmw.at<float>(2,1), Rmw.at<float>(2,2), tmw.at<float>(2),
		   0,                   0,                   0,                1);
	
	Tmr = Tmw*Twr;
	Tmr_gpu.upload(Tmr);

	// TODO check this carefully, why isn't dataContainer.step == rows*cols?
	updateCostVolumeCaller( (float*)K.data, (float*)Kinv.data, (float*)Tmr_gpu.data,
							rows, cols, currentImage.step,
							near, far, layers, rows*cols, 
							cdata, count,
							(float*)(Cmin.data), (float*)(Cmax.data), (float*)(CminIdx.data),
							(float4*)(referenceImage.data), (float4*)(currentImage.data), true);
}

void CostVolume::minimize_a(const cv::gpu::GpuMat& d, cv::gpu::GpuMat& a, float theta, float lambda)
{
	minimizeACaller(cdata, rows, cols, layers,
					(float*)a.data, (float*)d.data,
					(float*)CminIdx.data, 
					far, depthStep,
					theta, lambda);
}

CostVolume::~CostVolume()
{}
