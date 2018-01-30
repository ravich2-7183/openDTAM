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
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>

#include <dynamic_reconfigure/server.h>
#include <openDTAM/openDTAMConfig.h>

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
	int images_per_costvolume_; // TODO remove later after implementing auto mode
	int im_count_;

	// Regulariser properties
	float alpha_G_;
	float beta_G_;
	Regulariser regulariser_;

	// camera properties
	const int rows_, cols_;
	float fps_;
	Mat camera_matrix_;
	string transform_source_;

	// Optimization parameters
	GpuMat a_, d_;
	float theta_start_;
	float theta_min_;
	float theta_step_;
    float huber_epsilon_;
	float lambda_;
	int n_iters_;

	// ros
	ros::NodeHandle nh_;
	ros::Subscriber sub_;
	tf::TransformListener tf_listener_;
	tf::StampedTransform transform_;
	cv_bridge::CvImagePtr input_bridge_;
	std_msgs::Bool img_processed_msg_;
	ros::Publisher img_processed_pub_;
	ros::Publisher depth_pub_;
	ros::Publisher rgb_pub_;

	bool pause_execution_;

	// point cloud viewer
	bool updating_pointcloud_;
	boost::mutex update_pc_mutex_;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_;

public:
	DenseMapper(const cv::FileStorage& settings_file, bool pause_execution);

	void receiveImageStream();
	void processImage(const sensor_msgs::ImageConstPtr& image_msg);
	void getDepth(Mat& depth_map);
	void createPointCloud();
	void showPointCloud();
	void dynamicReconfigCallback(openDTAM::openDTAMConfig &config, uint32_t level);
	void dynamicReconfigThread();
	void publishDepthRGBImages();
	void optimize(int num_iters);
}; // class DepthEstimator
