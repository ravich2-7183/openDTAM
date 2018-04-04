#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <tf/tf.h>

#include "../../CostVolume.hpp"
#include "testTexture.cuh"

using namespace std;
using namespace cv;
using namespace cv::gpu;

void tfRead_cw(ifstream& poses_file, int index, Mat& Rcw, Mat& tcw)
{
	// using namespace std;
	// using namespace cv;

	poses_file.seekg(0);

	char line[256];
	size_t found;
	float pose[6];

	int idx;
	string line_str;
	do {
		poses_file.getline(line, 256);
		line_str = line;
		found = line_str.find_first_of(",");
		idx = stoi(line_str.substr(0, found));
	} while(idx != index);
	
	for(int i=0; i<6; i++) {
		size_t prev_found = found;
		found = line_str.find_first_of(",\n", found+1);
		pose[i] = stof(line_str.substr(prev_found+1, found));
		
		// cout << pose[i] << endl;
	}
	
	// DONE: check correctness of pose read-in
	// verified poses by reading the same from python and checking
	// setRPY uses 'sxyz' euler angle notation
	tf::Transform transform;
	transform.setOrigin(tf::Vector3(pose[0], pose[1], pose[2]));
	tf::Quaternion q;
	q.setRPY(pose[3], pose[4], pose[5]);
	transform.setRotation(q);

	tf::Matrix3x3 R = transform.getBasis();
	tf::Vector3   t = transform.getOrigin();

	cv::Mat Rwc = (Mat_<float>(3,3) <<
				   R[0].x(), R[0].y(), R[0].z(),
				   R[1].x(), R[1].y(), R[1].z(),
				   R[2].x(), R[2].y(), R[2].z());

	cv::Mat twc = (Mat_<float>(3,1) <<
				   t.x(),
				   t.y(),
				   t.z());

	Rcw =  Rwc.t();
	tcw = -Rwc.t()*twc;
}

void matPrint(Mat& a)
{
	cout << endl;
    for(int i=0; i<a.rows; i++) {
		for(int j=0; j<a.cols; j++) {
			cout << setw(10) << a.at<float>(i,j) << ", ";
		}
		cout << endl;
	}
	cout << endl;
}

// from: https://stackoverflow.com/questions/32332920/efficiently-load-a-large-mat-into-memory-in-opencv/32357875
void matWrite(const string& filename, const Mat& mat)
{
    ofstream fs(filename, fstream::binary);
	ofstream fs_info(filename + ".info");

    // Header
	fs_info << mat.rows << endl;
	fs_info << mat.cols << endl;
	fs_info << mat.type() << endl;
	fs_info << mat.channels() << endl;

    // Data
    if(mat.isContinuous()) {
        fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    }
    else {
        int rowsz = CV_ELEM_SIZE(mat.type()) * mat.cols;
        for (int r = 0; r < mat.rows; ++r) {
            fs.write(mat.ptr<char>(r), rowsz);
        }
    }
}

Mat matRead(const string& filename)
{
    ifstream fs(filename, fstream::binary);
	ifstream fs_info(filename + ".info");

    // Header
    int rows, cols, type, channels;
	fs_info >> rows;
	fs_info >> cols;
	fs_info >> type;
	fs_info >> channels;

    // Data
    Mat mat(rows, cols, type);
    fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

    return mat;
}

int main(int argc, char** argv) {
	string      settings_filename = argv[1];
	string		img_dir			  = argv[2];
	string		depth_dir		  = argv[3];
	string		poses_filename	  = argv[4];
	string		reference		  = argv[5]; 
	string		other			  = argv[6]; 

	cout << "\n img_dir        = "	<<	img_dir;
	cout << "\n poses_filename = "	<<	poses_filename;
	cout << "\n reference      = "	<<	reference;
	cout << "\n other          = "	<<	other;
	cout << endl;

	// read images
	Mat im_ref, im_other;
    im_ref   = imread(img_dir + "/" + reference + ".png", CV_LOAD_IMAGE_COLOR);
	im_other = imread(img_dir + "/" + other + ".png", CV_LOAD_IMAGE_COLOR);
	im_ref.convertTo(im_ref, CV_32FC3, 1.0/255.0);
	cvtColor(im_ref, im_ref, CV_BGR2RGBA);
	im_other.convertTo(im_other, CV_32FC3, 1.0/255.0);
	cvtColor(im_other, im_other, CV_BGR2RGBA);

	cout << img_dir + "/" + reference + ".png" << endl;
	cout << img_dir + "/" + other + ".png" << endl;

	// read and output ground truth depth image to bin file
	Mat im_exr, ground_depth, channels[3];
	im_exr = imread(depth_dir + "/" + reference + ".exr", CV_LOAD_IMAGE_UNCHANGED);
	split(im_exr, channels);
	ground_depth = channels[0];
	matWrite("./ground_depth.bin", ground_depth);

	// read tf
	ifstream poses_file(poses_filename);
	Mat Rrw = Mat::zeros(3,3, CV_32F);
	Mat Rmw = Mat::zeros(3,3, CV_32F);
	Mat trw = Mat::zeros(3,1, CV_32F);
	Mat tmw = Mat::zeros(3,1, CV_32F);
	tfRead_cw(poses_file, stoi(reference), Rrw, trw);
	tfRead_cw(poses_file, stoi(other)    , Rmw, tmw);
	poses_file.close();

	matPrint(Rrw);
	matPrint(trw);
	matPrint(Rmw);
	matPrint(tmw);

	// compute costvolume and output to file
	cv::FileStorage settings_file(settings_filename, cv::FileStorage::READ);
	
	int rows(settings_file["camera.rows"]);
	int cols(settings_file["camera.cols"]);
	int layers(settings_file["costvolume.layers"]);
	float near(settings_file["costvolume.near_inverse_distance"]);
	float far(settings_file["costvolume.far_inverse_distance"]);

	float fx, fy, cx, cy;
	fx = settings_file["camera.fx"];
	fy = settings_file["camera.fy"];
	cx = settings_file["camera.cx"];
	cy = settings_file["camera.cy"];

	Mat camera_matrix = (Mat_<float>(3,3) <<    fx,  0.0,  cx,
					                           0.0,   fy,  cy,
                                               0.0,  0.0, 1.0);

	CostVolume costvolume(rows, cols, layers, near, far, camera_matrix);

	costvolume.setReferenceImage(im_ref, Rrw, trw);
	costvolume.updateCost(im_other, Rmw, tmw);

	Mat costvolume_cpu;
	costvolume.cost_data.download(costvolume_cpu);
	matWrite("./costvolume.bin", costvolume_cpu);

	Mat inv_depth;
	costvolume.CminIdx.download(inv_depth);
	matWrite("./inv_depth.bin", inv_depth);

	// test texture
    cv::gpu::GpuMat current_image_color, texture_output_image;
	cv::gpu::createContinuous(rows, cols, CV_32FC4, current_image_color);
	cv::gpu::createContinuous(rows, cols, CV_32FC4, texture_output_image);
	current_image_color = 0;
	texture_output_image = 0;
	current_image_color.upload(im_other);
	testTextureCaller(rows, cols, current_image_color.step, (float4*)(current_image_color.data), (float4*)(texture_output_image.data));
	
	Mat texture_output_cpu, tex_channels[4];
	texture_output_image.download(texture_output_cpu);
	split(texture_output_cpu, tex_channels);

	matWrite("./texture_output_r.bin", tex_channels[0]);
	matWrite("./texture_output_g.bin", tex_channels[1]);
	matWrite("./texture_output_b.bin", tex_channels[2]);
}
