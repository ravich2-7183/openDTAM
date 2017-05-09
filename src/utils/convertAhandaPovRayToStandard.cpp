// This file converts the file format used at http://www.doc.ic.ac.uk/~ahanda/HighFrameRateTracking/downloads.html
// into the standard [R|T] world -> camera format used by OpenCV
// It is based on a file they provided there, but makes the world coordinate system right handed, with z up,
// x right, and y forward.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>

using namespace cv;
using namespace std;

static void readPoint3d(char *readlinedata, Point3d& point)
{
    istringstream iss;
    string point_str(readlinedata);
    
    point_str = point_str.substr(point_str.find("= [")+3);
    point_str = point_str.substr(0,point_str.find("]"));
    
    iss.str(point_str);
    iss >> point.x ;
    iss.ignore(1,',');
    iss >> point.z ;
    iss.ignore(1,',') ;
    iss >> point.y;
    iss.ignore(1,',');
}

void convertAhandaPovRayToStandard(const char * filepath,
                                   int imageNumber,
                                   Mat& R, Mat& T)
{
    char text_file_name[600];
    sprintf(text_file_name, "%s/scene_%03d.txt", filepath, imageNumber);
    // cout << "text_file_name = " << text_file_name << endl;

    ifstream cam_pars_file(text_file_name);
    if(!cam_pars_file.is_open()) {
        cerr<<"Failed to open param file, check location of sample trajectory!"<<endl;
        exit(1);
    }

    char readlinedata[300];
    Point3d direction, upvector, posvector;
    while(!cam_pars_file.eof()) {
        cam_pars_file.getline(readlinedata, 300);
        
        if(strstr(readlinedata, "cam_dir") != NULL)
            readPoint3d(readlinedata, direction);
        if(strstr(readlinedata, "cam_up")  != NULL)
            readPoint3d(readlinedata, upvector);
        if(strstr(readlinedata, "cam_pos") != NULL)
            readPoint3d(readlinedata, posvector);
    }

    R        = Mat(3, 3, CV_64F);
    R.row(0) = Mat(direction.cross(upvector)).t();
    R.row(1) = Mat(-upvector).t();
    R.row(2) = Mat(direction).t();
    T        = -R*Mat(posvector);
}
