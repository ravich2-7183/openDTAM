#include "DenseMapper.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;

DenseMapper::DenseMapper(const cv::FileStorage& settings_file, bool pause_execution) :
	nh_(),
	rows_(settings_file["camera.rows"]),
	cols_(settings_file["camera.cols"]),
	fps_(settings_file["camera.fps"]),
	transform_source_(settings_file["camera.transform_source"]),
	layers_(settings_file["costvolume.layers"]),
	near_(settings_file["costvolume.near_inverse_distance"]),
	far_(settings_file["costvolume.far_inverse_distance"]),
	images_per_costvolume_(settings_file["costvolume.images_per_costvolume"]),
	alpha_G_(settings_file["regulariser.alpha_G"]),
	beta_G_(settings_file["regulariser.beta_G"]),
	theta_start_(settings_file["optimizer.theta_start"]),
	theta_min_(settings_file["optimizer.theta_min"]),
	theta_step_(settings_file["optimizer.theta_step"]),
    huber_epsilon_(settings_file["optimizer.epsilon"]),
	lambda_(settings_file["optimizer.lambda"]),
	n_iters_(settings_file["optimizer.n_iters"]),
	pause_execution_(pause_execution),
	point_cloud_ptr_(new pcl::PointCloud<pcl::PointXYZRGB>())
{
	float fx, fy, cx, cy;
	fx = settings_file["camera.fx"];
	fy = settings_file["camera.fy"];
	cx = settings_file["camera.cx"];
	cy = settings_file["camera.cy"];

	// setup camera matrix
	camera_matrix_ = (Mat_<float>(3,3) <<    fx,  0.0,  cx,
                                            0.0,   fy,  cy,
                                            0.0,  0.0, 1.0);

	// TODO instead of relying on the default copy ctor, should i use a pointer and the new operator instead?
	CostVolume costvolume_tmp(rows_, cols_, layers_, near_, far_, camera_matrix_);
	costvolume_ = costvolume_tmp;

	cout << "In DenseMapper ctor" << endl;
	cout << "rows = " << costvolume_.rows << endl;
	cout << "cols = " << costvolume_.cols << endl;
	cout << "layers = " << costvolume_.layers << endl;
	cout << "imagesPerCostVolume = " << images_per_costvolume_ << endl;

	Regulariser regulariser_tmp(rows_, cols_, alpha_G_, beta_G_);
	regulariser_ = regulariser_tmp;

	// allocate space for a_ and d_
	cv::gpu::createContinuous(rows_, cols_, CV_32FC1, a_);
	cv::gpu::createContinuous(rows_, cols_, CV_32FC1, d_);

	img_processed_pub_ = nh_.advertise<std_msgs::Bool>("img_processed", 1000);

	depth_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/depth/image_raw", 10);
	rgb_pub_   = nh_.advertise<sensor_msgs::Image>("/camera/rgb/image_color", 10);
}

void DenseMapper::receiveImageStream()
{
	sub_ = nh_.subscribe("/camera/image_raw", 1, &DenseMapper::processImage, this);
	ros::spin();
}

void DenseMapper::processImage(const sensor_msgs::ImageConstPtr& image_msg)
{
	Mat image_; // TODO rename variable
	try {
		input_bridge_ = cv_bridge::toCvCopy(image_msg); // TODO: use cv_bridge::toCvShare instead?
		image_ = input_bridge_->image; // TODO: is this copy required?
		image_.convertTo(image_, CV_32FC3, 1.0/255.0); // float image [0-1] // TODO: make this more generic
		cvtColor(image_, image_, CV_RGB2RGBA); // TODO check if image is in BGR format. RGBA for better memory alignment. 

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
		tf_listener_.waitForTransform("/" + transform_source_ + "/Camera", "/" + transform_source_ + "/World",
									  acquisition_time, timeout);
		tf_listener_.lookupTransform("/" + transform_source_ + "/Camera", "/" + transform_source_ + "/World",
									 acquisition_time, transform_);

		// TODO debug lines
		cout << "Received transform with time stamp: " << transform_.stamp_ << endl;
	}
	catch (tf::TransformException& ex) {
		ROS_WARN("[DenseMapper] TF exception: \n%s", ex.what());

		tf_listener_.lookupTransform("/" + transform_source_ + "/Camera", "/" + transform_source_ + "/World",
									 ros::Time(0), transform_);

		// TODO debug lines
		cout << "Received delayed transform with time stamp: " << transform_.stamp_ << endl;
	}

	tf::Matrix3x3 R = transform_.getBasis();
	tf::Vector3   t = transform_.getOrigin();

	cv::Mat Rcw = (Mat_<float>(3,3) <<
				   R[0].x(), R[0].y(), R[0].z(),
				   R[1].x(), R[1].y(), R[1].z(),
				   R[2].x(), R[2].y(), R[2].z());

	cv::Mat tcw = (Mat_<float>(3,1) <<
				   t.x(),
				   t.y(),
				   t.z());

	if(costvolume_.count_ == 0) {
		cout << "resetting reference image" << endl;

		costvolume_.setReferenceImage(image_, Rcw, tcw);
		regulariser_.initialize(costvolume_.reference_image_gray_);

		// TODO debug lines
		namedWindow("g", WINDOW_AUTOSIZE);
		Mat gImg;
		gImg.create(rows_, cols_, CV_32FC1);
		regulariser_.g_.download(gImg);
		// aImg *= (1.0f/costvolume_.near);
		imshow("g", gImg); waitKey(10);
		
		costvolume_.count_++;

		// img_processed_pub_.publish(img_processed_msg_);
		return;
	}
	
	cout << "count_ = " << costvolume_.count_ << endl;
	cout << "updating cost volume" << endl;

	costvolume_.updateCost(image_, Rcw, tcw);

	costvolume_.CminIdx.copyTo(d_);
	costvolume_.CminIdx.copyTo(a_);

	if(costvolume_.count_ >= images_per_costvolume_ + 1) { 
		this->optimize(0); // 0: fully optimize

		createPointCloud();
		publishDepthRGBImages();

		costvolume_.reset();

		if(!pause_execution_)
			img_processed_pub_.publish(img_processed_msg_);
	}
	else {
		this->optimize(n_iters_);

		createPointCloud();
		publishDepthRGBImages();

		// img_processed_pub_.publish(img_processed_msg_);
	}
}

void DenseMapper::getDepth(Mat& depth_map)
{
	d_.download(depth_map);
}

void DenseMapper::createPointCloud()
{
	boost::mutex::scoped_lock update_lock(update_pc_mutex_);
	updating_pointcloud_ = true;

	Mat inv_depth, Kinv, reference_image;

	d_.download(inv_depth);
	costvolume_.Kinv.download(Kinv);
	costvolume_.reference_image_color_.download(reference_image);

	// create point cloud from depth map
	point_cloud_ptr_->points.clear();
	for(int v=0; v<rows_; v++) {
		for(int u=0; u<cols_; u++) {
			pcl::PointXYZRGB point;

			point.z = 1.0/inv_depth.at<float>(v,u);
			point.x = (Kinv.at<float>(0,0)*u + Kinv.at<float>(0,2)) * point.z;
			point.y = (Kinv.at<float>(1,1)*v + Kinv.at<float>(1,2)) * point.z;
			point.b = static_cast<uint8_t>(reference_image.at<cv::Vec4f>(v,u)[0] * 255);
			point.g = static_cast<uint8_t>(reference_image.at<cv::Vec4f>(v,u)[1] * 255);
			point.r = static_cast<uint8_t>(reference_image.at<cv::Vec4f>(v,u)[2] * 255);

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
	viewer->setCameraPosition(-0.0067756, -0.0685564, -0.462478,
							           0,          0,         1,
							  -0.0105255, -0.9988450,  0.0468715);
	viewer->setCameraClipDistances(0.0186334, 18.6334);
	viewer->setCameraFieldOfView(0.8575);

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

void DenseMapper::dynamicReconfigCallback(openDTAM::openDTAMConfig &config, uint32_t level)
{
	ROS_INFO("Processing reconfigure request.");

	images_per_costvolume_ =         config.costvolume_images;
//  layers_                =         config.costvolume_layers;
	near_                  =         1 / config.near_distance;
	far_                   =         1 / config.far_distance;
	alpha_G_               =         config.alpha_G;
	beta_G_                =         config.beta_G;
	theta_start_           =         config.theta_start;
	theta_min_             =         config.theta_min;
	theta_step_            =         config.theta_step;
    huber_epsilon_         =         config.huber_epsilon;
	lambda_                =         config.lambda;
	n_iters_               =         config.n_iters;

	cout <<	"images_per_costvolume_ = " << images_per_costvolume_ << endl;
	cout <<	"near_                  = " << config.near_distance   << endl;
	cout <<	"far_                   = " << config.far_distance    << endl;
	cout <<	"alpha_G_               = " << alpha_G_               << endl;
	cout <<	"beta_G_                = " << beta_G_                << endl;
	cout <<	"theta_start_           = " << theta_start_           << endl;
	cout <<	"theta_min_             = " << theta_min_             << endl;
	cout <<	"theta_step_            = " << theta_step_            << endl;
    cout <<	"epsilon_               = " << huber_epsilon_         << endl;
	cout <<	"lambda_                = " << lambda_                << endl;
	cout <<	"n_iters_               = " << n_iters_               << endl;
}

void DenseMapper::publishDepthRGBImages()
{
	// publish depth image
	Mat inv_depth;
	cv_bridge::CvImage cv_depth_image;
	sensor_msgs::Image ros_depth_image;

	d_.download(inv_depth);

	Mat ONES = Mat::ones(rows_, cols_, CV_32FC1);
	Mat depth_mm_F = ONES.mul(1/inv_depth);
	Mat depth_mm_U;
	depth_mm_F.convertTo(depth_mm_U, CV_16UC1, 1000); // TODO make sure that this does the right thing
	cv_depth_image.image = depth_mm_U;

	cv_depth_image.toImageMsg(ros_depth_image);
	ros_depth_image.header.stamp = ros::Time::now();
	ros_depth_image.encoding = "16UC1";
	depth_pub_.publish(ros_depth_image);

	// publish rgb image
	Mat rgb_image;
	cv_bridge::CvImage cv_rgb_image;
	sensor_msgs::Image ros_rgb_image;

	costvolume_.reference_image_color_.download(rgb_image);
	rgb_image.convertTo(rgb_image, CV_8UC4, 255);
	cvtColor(rgb_image, rgb_image, CV_RGBA2BGR);
	cv_rgb_image.image = rgb_image;

	cv_rgb_image.toImageMsg(ros_rgb_image);
	ros_rgb_image.encoding = "bgr8";
	ros_rgb_image.header.stamp = ros::Time::now();
	rgb_pub_.publish(ros_rgb_image);
}

void DenseMapper::optimize(int num_iters)
{
	int n = 1;
	
	if(num_iters == 0) {
		float theta = theta_start_;
		while(theta > theta_min_) {
			regulariser_.update_q_d(a_, d_, huber_epsilon_, theta);

			costvolume_.minimize_a(d_, a_, theta, lambda_); // point wise search for a[] that minimizes Eaux

			float beta = (theta > 1e-3)? 1e-3 : 1e-4;
			theta *= (1-beta*n);
			n++;
		}
	}
	else {
		float theta = theta_start_;
		while(n < num_iters) {
			regulariser_.update_q_d(a_, d_, huber_epsilon_, theta);

			costvolume_.minimize_a(d_, a_, theta, lambda_); // point wise search for a[] that minimizes Eaux

			float beta = (theta > 1e-3)? 1e-3 : 1e-4;
			theta *= (1-beta*n);
			n++;
		}
	}
}
