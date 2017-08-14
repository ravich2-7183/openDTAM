#include "CostVolume/CostVolume.hpp"
#include "Regulariser/Regulariser.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/device/common.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/image_encodings.h>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace cv;
using namespace cv::gpu;

class DenseMapper
{
public:
	// CostVolume properties
	int layers_;
	float near_;
	float far_;
	CostVolume costvolume_;
	int imagesPerCostVolume_; // TODO remove later after implementing auto mode
	int im_count_;

	// Regulariser properties
	float alpha_G_;
	float beta_G_;
	Regulariser regulariser_;

	// camera properties
	const int rows_, cols_;
	double fps_;
	Mat camera_matrix_;

	// Optimization parameters
	GpuMat a_, d_;
	const float theta_start_;
	const float theta_min_;
	const float theta_step_;
	const float epsilon_;
	const float lambda_;

	// ros
	ros::NodeHandle nh_;
	ros::Subscriber sub_;
	tf::TransformListener tf_listener_;
	tf::StampedTransform transform_;
	cv_bridge::CvImagePtr input_bridge_;

	// point cloud viewer
	bool updating_pointcloud_;
	boost::mutex update_pc_mutex_;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_;

public:
	DenseMapper(const cv::FileStorage& settings_file);

	void receiveImageStream();
	void processImage(const sensor_msgs::ImageConstPtr& image_msg);
	void resetCostVolume(const cv::Mat& image, const cv::Mat& Rrw, const cv::Mat& trw);
	void updateCostVolume(const Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw);
	void getDepth(Mat& depth_map);
	void createPointCloud();
	void showPointCloud();
	void optimize();
}; // class DepthEstimator
