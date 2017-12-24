#ifndef COSTVOLUME_HPP
#define COSTVOLUME_HPP

#include <opencv2/gpu/gpu.hpp>

class CostVolume
{
public:
	int rows;
	int cols;
	int layers;
	float near;
	float far;
	float depthStep;

	cv::gpu::GpuMat K; // Note: should be in OpenCV format
	cv::gpu::GpuMat Kinv; // TODO ensure they are contiguous

	cv::Mat Rwr, twr;
	cv::Mat Tmw, Twr, Tmr;
	cv::gpu::GpuMat Tmr_gpu;

    cv::gpu::GpuMat reference_image_color_;
    cv::gpu::GpuMat reference_image_gray_;
    cv::gpu::GpuMat current_image_color_;
    cv::gpu::GpuMat current_image_gray_;
	cv::gpu::GpuMat Cmin; // TODO unnecessary variable
	cv::gpu::GpuMat Cmax; // TODO unnecessary variable
	cv::gpu::GpuMat CminIdx; // TODO a better name: d_Cmin?

    cv::gpu::GpuMat cost_data; // TODO rename this to Cdata for consistency
	float *cdata; // TODO unnecessary variable

    float count_;
    
	CostVolume() {};
	CostVolume(float _rows, float _cols, float _layers, float _near, float _far, const cv::Mat& Kcpu);
	~CostVolume();
  
	void reset();
	void setReferenceImage(const cv::Mat& reference_image, const cv::Mat& Rrw, const cv::Mat& trw);
	void updateCost(const cv::Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw);
	void minimize_a(const cv::gpu::GpuMat& d, cv::gpu::GpuMat& a, float theta, float lambda);
};

#endif // COSTVOLUME_HPP
