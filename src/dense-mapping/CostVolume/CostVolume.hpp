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

	cv::gpu::GpuMat referenceImage;
	cv::gpu::GpuMat referenceImageGray;
	cv::gpu::GpuMat currentImage;
	cv::gpu::GpuMat Cmin; // TODO unnecessary variable
	cv::gpu::GpuMat Cmax; // TODO unnecessary variable
	cv::gpu::GpuMat CminIdx; // TODO a better name: d_Cmin?

	cv::gpu::GpuMat dataContainer; // TODO rename this to Cdata for consistency
	float *cdata; // TODO unnecessary variable

	float count;
    
	CostVolume() {};
	CostVolume(float _rows, float _cols, float _layers, float _near, float _far);
	~CostVolume();
  
	void reset(const cv::Mat& image, const cv::Mat& Kcpu, const cv::Mat& Rrw, const cv::Mat& trw);
	void updateCost(const cv::Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw);
	void minimize_a(const cv::gpu::GpuMat& d, cv::gpu::GpuMat& a, float theta, float lambda);
  
private:
	void checkInputs(const cv::Mat& R, const cv::Mat& t,
					 const cv::Mat& image,
					 const cv::Mat& cameraMatrix);
};

#endif // COSTVOLUME_HPP
