#include <opencv2/core/core.hpp>
#include <stdio.h>

#include "DepthEstimator.hpp"
#include "../utils/convertAhandaPovRayToStandard.h"

//A test program to make the mapper run
using namespace cv;
using namespace cv::gpu;
using namespace std;

void App_main(const cv::FileStorage& settings_file)
{
	const int numImg = 30;
	const int imagesPerCostVolume = settings_file["imagesPerCostVolume"];

	cv::namedWindow("inverse_depth", WINDOW_AUTOSIZE);

	char filename[500];
	Mat image;
	Mat Rcw, tcw;

	DepthEstimator depthEstimator(settings_file);

	Mat depth_map;

	for(int i=0; i < numImg; i++) {
		sprintf(filename, "../Trajectory_30_seconds/scene_%03d.png", i);
		printf("Opening: %s \n", filename);
		image = imread(filename, -1);

		image.convertTo(image, CV_32FC3, 1.0/65535.0); // Ahanda images are CV_16UC3, which range from 0-65535
		cvtColor(image, image, CV_RGB2RGBA);

		convertAhandaPovRayToStandard("../Trajectory_30_seconds", i, Rcw, tcw);

		if(i % imagesPerCostVolume == 0) {
			if(i != 0) {
				depthEstimator.optimize();

				depthEstimator.getDepth(depth_map);
				imshow("inverse_depth", depth_map);
				waitKey(30);
			}
			
			depthEstimator.resetCostVolume(image, Rcw, tcw);
			cout << "CostVolume reset" << endl;
		}
		else {
			depthEstimator.updateCostVolume(image, Rcw, tcw);
		}
	}
    cv::destroyWindow("inverse_depth");
}

int main(int argc, char** argv) {
	if(argc < 2) {
		cout << "\nUsage: executable_name path/to/settings_file \n";
		exit(0);
	}

	string settings_filename = argv[1];
	cv::FileStorage settings_file(settings_filename, cv::FileStorage::READ);

	App_main(settings_file);

	return 0;
}
