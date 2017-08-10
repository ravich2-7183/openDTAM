#include <string>
#include <iostream>

#include "DenseMapper.hpp"

// test program for the mapper, using camera poses from orb_slam

using namespace std;

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
	
	boost::thread dense_mapper_thread(&DenseMapper::receiveImageStream, &dense_mapper);
	boost::thread pointcloud_visualiser_thread(&DenseMapper::showPointCloud, &dense_mapper);

	dense_mapper_thread.join();
	pointcloud_visualiser_thread.join();
	
	ros::shutdown();
	return 0;
}
