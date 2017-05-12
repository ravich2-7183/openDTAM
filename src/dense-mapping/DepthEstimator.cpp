#include "DepthEstimator.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

DepthEstimator::DepthEstimator(const cv::FileStorage& settings_file) :
	rows(settings_file["Camera.rows"]),
	cols(settings_file["Camera.cols"]),
	layers(settings_file["costVolumeLayers"]),
	near(settings_file["nearInverseDistance"]),
	far(settings_file["farInverseDistance"]),
	alphaG(settings_file["alphaG"]),
	betaG(settings_file["betaG"]),
	theta_start(settings_file["theta_start"]),
	theta_min(settings_file["theta_min"]),
	theta_step(settings_file["theta_step"]),
	epsilon(settings_file["epsilon"]),
	lambda(settings_file["lambda"])
{
	// TODO instead of relying on the default copy ctor, should i use a pointer and the new operator instead?
	CostVolume costvolume_tmp(rows, cols, layers, near, far);
	costvolume = costvolume_tmp;

	cout << "In DepthEstimator ctor" << endl;
	cout << "rows = " << costvolume.rows << endl;
	cout << "cols = " << costvolume.cols << endl;
	cout << "layers = " << costvolume.layers << endl;

	Regulariser regulariser_tmp(rows, cols, alphaG, betaG);
	regulariser = regulariser_tmp;

	double fx, fy, cx, cy;
	fx = settings_file["Camera.fx"];
	fy = settings_file["Camera.fy"];
	cx = settings_file["Camera.cx"];
	cy = settings_file["Camera.cy"];

	// setup camera matrix
	camera_matrix = (Mat_<double>(3,3) <<    fx,  0.0,  cx,
											0.0,   fy,  cy,
											0.0,  0.0, 1.0);
	
	cv::gpu::createContinuous(rows, cols, CV_32FC1, a);
	cv::gpu::createContinuous(rows, cols, CV_32FC1, d);
}

void DepthEstimator::resetCostVolume(const cv::Mat& image, const cv::Mat& Rrw, const cv::Mat& trw)
{
	costvolume.reset(image, camera_matrix, Rrw, trw);
	regulariser.initialize(costvolume.referenceImageGray);
}

void DepthEstimator::updateCostVolume(const Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw)
{
	costvolume.updateCost(image, Rmw, tmw);
}

void DepthEstimator::getDepth(Mat& depth_map)
{
	d.download(depth_map);
}

void DepthEstimator::optimize()
{
	// Initialize a, d
	costvolume.CminIdx.copyTo(a);
	costvolume.CminIdx.copyTo(d);

	// TODO remove debug displays
	cout << "Optimization start: --------" << endl;
	Mat aImg, dImg; //
	aImg.create(rows, cols, CV_32FC1); 	dImg.create(rows, cols, CV_32FC1);
	a.download(aImg);	d.download(dImg);
	namedWindow("a", WINDOW_AUTOSIZE);
	imshow("a", aImg*(1/costvolume.near)); 	waitKey(10);
	namedWindow("d", WINDOW_AUTOSIZE); 
	imshow("d", dImg*(1/costvolume.near)); 	waitKey(10);

	unsigned int n = 1;
	float theta = theta_start;
	// while(theta > theta_min) {
	while(n < 100) {
		// step 1.
		regulariser.update_q_d(a, d, epsilon, theta); // update_q_d must be called before minimize_a or else no progress will be made

		// TODO debug lines
		d.download(dImg);
		imshow("d", dImg*(1/costvolume.near));
		waitKey(10);

		// step 2.
		costvolume.minimize_a(d, a, theta, lambda); // point wise search for a[] that minimizes Eaux
		
		// TODO debug lines
		a.download(aImg);
		imshow("a", aImg*(1/costvolume.near));
		waitKey(10);

		// step 3.
		float beta = (theta > 1e-3)? 1e-3 : 1e-4;
		theta *= (1-beta*n);
		n++;

	}
}

