#include "DenseMapper.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;

DenseMapper::DenseMapper(const cv::FileStorage& settings_file) :
	nh_(),
	rows_(settings_file["camera.rows"]),
	cols_(settings_file["camera.cols"]),
	fps_(settings_file["camera.fps"]),
	layers_(settings_file["costvolume.layers"]),
	near_(settings_file["costvolume.near_inverse_distance"]),
	far_(settings_file["costvolume.far_inverse_distance"]),
	imagesPerCostVolume_(settings_file["costvolume.images_per_costvolume"]),
	alpha_G_(settings_file["regulariser.alpha_G"]),
	beta_G_(settings_file["regulariser.beta_G"]),
	theta_start_(settings_file["optimizer.theta_start"]),
	theta_min_(settings_file["optimizer.theta_min"]),
	theta_step_(settings_file["optimizer.theta_step"]),
	epsilon_(settings_file["optimizer.epsilon"]),
	lambda_(settings_file["optimizer.lambda"]),
	point_cloud_ptr_(new pcl::PointCloud<pcl::PointXYZRGB>())
{
	// TODO instead of relying on the default copy ctor, should i use a pointer and the new operator instead?
	CostVolume costvolume_tmp(rows_, cols_, layers_, near_, far_);
	costvolume_ = costvolume_tmp;
	im_count_ = 0;

	cout << "In DenseMapper ctor" << endl;
	cout << "rows = " << costvolume_.rows << endl;
	cout << "cols = " << costvolume_.cols << endl;
	cout << "layers = " << costvolume_.layers << endl;
	cout << "imagesPerCostVolume = " << imagesPerCostVolume_ << endl;

	Regulariser regulariser_tmp(rows_, cols_, alpha_G_, beta_G_);
	regulariser_ = regulariser_tmp;

	double fx, fy, cx, cy;
	fx = settings_file["camera.fx"];
	fy = settings_file["camera.fy"];
	cx = settings_file["camera.cx"];
	cy = settings_file["camera.cy"];

	// setup camera matrix
	camera_matrix_ = (Mat_<double>(3,3) <<    fx,  0.0,  cx,
											0.0,   fy,  cy,
											0.0,  0.0, 1.0);

	cv::gpu::createContinuous(rows_, cols_, CV_32FC1, a_);
	cv::gpu::createContinuous(rows_, cols_, CV_32FC1, d_);
}

void DenseMapper::receiveImageStream()
{
    sub_ = nh_.subscribe("/camera/image_raw", 5, &DenseMapper::processImage, this);
	ros::spin();
}

void DenseMapper::processImage(const sensor_msgs::ImageConstPtr& image_msg)
{
	Mat image_; // TODO rename variable
	try {
		input_bridge_ = cv_bridge::toCvCopy(image_msg); // TODO: use cv_bridge::toCvShare instead?
		image_ = input_bridge_->image; // TODO: is this copy required?
		image_.convertTo(image_, CV_32FC3, 1.0/255.0); // float image [0-1] // TODO: make this more generic
		cvtColor(image_, image_, CV_RGB2RGBA); // TODO check if image is in BGR format

		// TODO debug lines
		cout << "Received image with time stamp: " << image_msg->header.stamp << endl;
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

		// tf_listener_.lookupTransform("/ORB_SLAM/World", "/ORB_SLAM/Camera",
		// 							 ros::Time(0), transform_);

		// TODO debug lines
		cout << "Received transform with time stamp: " << transform_.stamp_ << endl;
	}
	catch (tf::TransformException& ex) {
		ROS_WARN("[DenseMapper] TF exception: \n%s", ex.what());
		return;
	}

	tf::Matrix3x3 R = transform_.getBasis();
	tf::Vector3   t = transform_.getOrigin();

	cv::Mat Rcw = (Mat_<double>(3,3) << 
				   R[0].x(), R[0].y(), R[0].z(),
				   R[1].x(), R[1].y(), R[1].z(),
				   R[2].x(), R[2].y(), R[2].z());
		
	cv::Mat tcw = (Mat_<double>(3,1) << 
				   t.x(),
				   t.y(),
				   t.z());

	// // TODO debug
	// cout << "Rcw = " << Rcw << endl;
	// cout << "tcw = " << tcw << endl;

	tcw = (-Rcw.t())*tcw;
	Rcw =  Rcw.t();

	if(im_count_ % imagesPerCostVolume_ == 0) { // TODO replace im_count_ with internal variable
		if(im_count_ != 0) {
			this->optimize();
		}

		this->resetCostVolume(image_, Rcw, tcw);
		cout << "CostVolume reset" << endl;
	}
	else {
		this->updateCostVolume(image_, Rcw, tcw);
		// TODO print message
	}

	im_count_++;
}

void DenseMapper::resetCostVolume(const cv::Mat& image, const cv::Mat& Rrw, const cv::Mat& trw)
{
	costvolume_.reset(image, camera_matrix_, Rrw, trw);
	regulariser_.initialize(costvolume_.referenceImageGray);
}

void DenseMapper::updateCostVolume(const Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw)
{
	costvolume_.updateCost(image, Rmw, tmw);
}

void DenseMapper::getDepth(Mat& depth_map)
{
	d_.download(depth_map);
}

void DenseMapper::createPointCloud()
{
	boost::mutex::scoped_lock update_lock(update_pc_mutex_);
	updating_pointcloud_ = true;

	Mat depth, Kinv, referenceImage;

	// depth.create(rows_, cols_, CV_32FC1);
	d_.download(depth);
	// depth = depth*(1/costvolume_.near);

	costvolume_.Kinv.download(Kinv);
	costvolume_.referenceImage.download(referenceImage);

	// create point cloud from depth map
	point_cloud_ptr_->points.clear();
	for(int v=0; v<rows_; v++) {
		for(int u=0; u<cols_; u++) {
			pcl::PointXYZRGB point;
			
			point.z = 1.0/depth.at<float>(v,u);
			point.x = (Kinv.at<double>(0,0)*u + Kinv.at<double>(0,2)) * point.z;
			point.y = (Kinv.at<double>(1,1)*v + Kinv.at<double>(1,2)) * point.z;
			point.b = static_cast<uint8_t>(referenceImage.at<cv::Vec4f>(v,u)[0] * 255);
			point.g = static_cast<uint8_t>(referenceImage.at<cv::Vec4f>(v,u)[1] * 255);
			point.r = static_cast<uint8_t>(referenceImage.at<cv::Vec4f>(v,u)[2] * 255);
			
			point_cloud_ptr_->points.push_back(point);
		}
	}
	
	update_lock.unlock();
}

void DenseMapper::showPointCloud()
{
	// Setup PCL 3D point cloud viewer and start thread
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Projected Depth Image"));

	point_cloud_ptr_->points.push_back(pcl::PointXYZRGB());
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr_);

	viewer->setBackgroundColor(0, 0, 0);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "projected_depth_image");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	
	while(!viewer->wasStopped()) {
		viewer->spinOnce (100);
		
		boost::mutex::scoped_lock update_lock(update_pc_mutex_);
		if(updating_pointcloud_) {
			if(!viewer->updatePointCloud(point_cloud_ptr_, "projected_depth_image"))
				viewer->addPointCloud(point_cloud_ptr_, rgb, "projected_depth_image");
			updating_pointcloud_ = false;
		}
		update_lock.unlock();

		// boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
}

void DenseMapper::optimize()
{
	// Initialize a, d
	costvolume_.CminIdx.copyTo(a_);
	costvolume_.CminIdx.copyTo(d_);

	// TODO debug lines
	cout << "Optimization start: --------" << endl;
	Mat aImg, dImg; //
	aImg.create(rows_, cols_, CV_32FC1);	dImg.create(rows_, cols_, CV_32FC1);
	namedWindow("a", WINDOW_AUTOSIZE);
	namedWindow("d", WINDOW_AUTOSIZE);

	a_.download(aImg);	d_.download(dImg);
	imshow("a", aImg*(1/costvolume_.near));		waitKey(10); // scale float image to lie in [0-1]
	imshow("d", dImg*(1/costvolume_.near));		waitKey(10);

	createPointCloud();

	unsigned int n = 1;
	float theta = theta_start_;
	while(theta > theta_min_) {
	// while(n < 100) {
		// step 1. update_q_d must be called before minimize_a or else no progress will be made
		regulariser_.update_q_d(a_, d_, epsilon_, theta);


		// step 2.
		costvolume_.minimize_a(d_, a_, theta, lambda_); // point wise search for a[] that minimizes Eaux


		// step 3.
		float beta = (theta > 1e-3)? 1e-3 : 1e-4;
		theta *= (1-beta*n);
		n++;
	}

	// TODO debug lines
	d_.download(dImg);
	imshow("d", dImg*(1/costvolume_.near));
	waitKey(10);

	// TODO debug lines
	a_.download(aImg);
	imshow("a", aImg*(1/costvolume_.near));
	waitKey(10);

	// TODO debug lines
	createPointCloud();
}
