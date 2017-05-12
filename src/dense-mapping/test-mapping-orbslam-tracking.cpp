#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/image_encodings.h>
#include <string>
#include <iostream>

#include "DepthEstimator.hpp"

// test program for the mapper, using camera poses from orb_slam

using namespace cv;
using namespace cv::gpu;
using namespace std;

class DenseMapper
{
	ros::NodeHandle nh_;
	ros::Subscriber sub_;
	tf::TransformListener tf_listener_;
	tf::StampedTransform transform_;

	int imagesPerCostVolume_;

	DepthEstimator depthEstimator_;

	// camera properties
	int im_count_;
	double fps_;
	cv_bridge::CvImagePtr input_bridge_;

public:
	DenseMapper(cv::FileStorage& settings_file) :
		nh_(),
		imagesPerCostVolume_(settings_file["imagesPerCostVolume"]),
		depthEstimator_(settings_file),
		fps_(settings_file["Camera.fps"])
	{
		im_count_ = 0;
	}

	void Run()
	{
		sub_ = nh_.subscribe("/camera/image_raw", 1, &DenseMapper::imageCb, this);
		ros::spin();
	}

	void imageCb(const sensor_msgs::ImageConstPtr& image_msg)
	{
		Mat image_;
		try {
			input_bridge_ = cv_bridge::toCvCopy(image_msg); // TODO: use cv_bridge::toCvShare instead?
			image_ = input_bridge_->image; // TODO: is this copy required?
			image_.convertTo(image_, CV_32FC3, 1.0/65535.0);
			cvtColor(image_, image_, CV_RGB2RGBA);
		}
		catch (cv_bridge::Exception& ex) {
			ROS_ERROR("[DenseMapper] Failed to convert image: \n%s", ex.what());
			return;
		}

		ros::Time acquisition_time = image_msg->header.stamp;
		ros::Duration timeout(1.0 / fps_);
		try {
			// TODO is the inverse transform the required one ???
			tf_listener_.waitForTransform("/ORB_SLAM/World", "/ORB_SLAM/Camera",
										  acquisition_time, timeout);
			tf_listener_.lookupTransform("/ORB_SLAM/World", "/ORB_SLAM/Camera",
										 acquisition_time, transform_);
		}
		catch (tf::TransformException& ex) {
			ROS_WARN("[DenseMapper] TF exception: \n%s", ex.what());
			return;
		}

		tf::Matrix3x3 R = transform_.getBasis();
		tf::Vector3   t = transform_.getOrigin();

		// DONE are the axes similarly aligned to the ahanda dataset? most likely yes.
		cv::Mat Rcw = (Mat_<double>(3,3) << 
					   R[0].x(), R[0].y(), R[0].z(),
					   R[1].x(), R[1].y(), R[1].z(),
					   R[2].x(), R[2].y(), R[2].z());
		
		cv::Mat tcw = (Mat_<double>(3,1) << 
					   t.x(),
					   t.y(),
					   t.z());

		tcw = (-Rcw.t())*tcw;
		Rcw =  Rcw.t();
		// Tcw *= 100.00; // TODO: See if this works

		if(im_count_ % imagesPerCostVolume_ == 0) {
			if(im_count_ != 0) {
				depthEstimator_.optimize();
			}

			depthEstimator_.resetCostVolume(image_, Rcw, tcw);
			cout << "CostVolume reset" << endl;
		}
		else {
			depthEstimator_.updateCostVolume(image_, Rcw, tcw);
		}

		im_count_++;
	} // DenseMapper::imageCb close
}; // class DenseMapper

int main(int argc, char** argv) {
	if(argc < 2) {
		cout << "\nUsage: executable_name path/to/settings_file \n";
		exit(0);
	}

	string settings_filename = argv[1];
	cv::FileStorage settings_file(settings_filename, cv::FileStorage::READ);

	ros::init(argc, argv, "dense_mapper");
	ros::start();

	DenseMapper dense_mapper(settings_file);
	boost::thread dense_mapper_thread(&DenseMapper::Run, &dense_mapper);
	dense_mapper_thread.join();

	ros::shutdown();
	return 0;
}
