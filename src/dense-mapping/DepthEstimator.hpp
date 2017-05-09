#include "CostVolume/CostVolume.hpp"
#include "Regulariser/Regulariser.hpp"

#include <opencv2/core/operations.hpp>
#include <opencv2/gpu/device/common.hpp>

using namespace cv;
using namespace cv::gpu;

class DepthEstimator
{
public:
	// CostVolume properties
	const int layers;
	const float near;
	const float far;
	CostVolume costvolume;

	// Regulariser properties
	float alphaG;
	float betaG;
	Regulariser regulariser;

	// camera properties
	const int rows, cols;
	Mat camera_matrix;

	// Optimization parameters
	GpuMat a, d;
	const float theta_start;
	const float theta_min;
	const float theta_step;
	const float epsilon;
	const float lambda;

public:
	DepthEstimator(const cv::FileStorage& settings_file);
	
	void resetCostVolume(const cv::Mat& image, const cv::Mat& Rrw, const cv::Mat& trw);
	void updateCostVolume(const Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw);
	void getDepth(Mat& depth_map);
	void optimize();
}; // class DepthEstimator
