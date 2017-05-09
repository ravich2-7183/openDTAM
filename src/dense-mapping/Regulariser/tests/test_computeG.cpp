#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <chrono>
#include <string>
#include <iostream>
#include <stdio.h>
#include "computeG.cuh"

using namespace cv;
using namespace cv::gpu;
using namespace std;

int _N = 10;

GpuMat _g1,_gx,_gy;
int _rows, _cols;

#define FLATALLOC(a,m) a.create(1,m.rows*m.cols, CV_32FC1); a = a.reshape(0,m.rows)
void allocate() {
  _g1.create(1,_rows*_cols, CV_32FC1);
  _g1=_g1.reshape(0,_rows);

  FLATALLOC(_gx,_g1);
  FLATALLOC(_gy,_g1);
}

void callComputeG(GpuMat& im){
  float* pp  = (float*) im.data;
  float* g1p = (float*)_g1.data;
  float* gxp = (float*)_gx.data;
  float* gyp = (float*)_gy.data;
  computeGCaller(pp,  g1p,  gxp,  gyp, _rows, _cols);
}

void callComputeGScharr(GpuMat& im, GpuMat& grad){
  float* im_data  = (float*) im.data;
  float* g = (float*) grad.data;

  computeGScharrCaller(im_data, g, im.cols, im.rows, im.step);
}

auto _startGPU = chrono::high_resolution_clock::now();
auto _endGPU   = chrono::high_resolution_clock::now();
void printTime(string codelabel, float N = 1.0f) {
  cout << codelabel
       << " ran in: \t\t"
       <<  (float)(chrono::duration_cast<chrono::microseconds>(_endGPU - _startGPU).count())/N
       << " usec"
       << endl;
}

Mat _disp_grad;
void showResult(GpuMat& result, string windowname) {
  result.download(_disp_grad);
  namedWindow(windowname, WINDOW_AUTOSIZE);
  imshow(windowname, _disp_grad);
  waitKey(0);
}

int main(){
  char filename[500];
  Mat im;
  sprintf(filename, "../../../../Trajectory_30_seconds/scene_%03d.png", 1);
  printf("Opening: %s \n", filename);
  imread(filename, -1).convertTo(im, CV_32FC3, 1.0/65535.0);
  cvtColor(im, im, CV_RGB2GRAY);

  namedWindow("input image", WINDOW_AUTOSIZE );
  imshow("input image", im );
  waitKey(0);

  _rows=im.rows;
  _cols=im.cols;

  CV_Assert((_rows % 32 == 0 && _cols % 32 == 0 && _cols >= 64));

  GpuMat img(im);

  // compute gradient using computeGScharr
  GpuMat grad(_rows, _cols, CV_32FC1);
  // TODO find a way to load kernels before hand, rather than just in time
  callComputeGScharr(img, grad); // force the kernel to get loaded first
  _startGPU = chrono::high_resolution_clock::now();
  {
    for(int i=1;i<=_N;i++)
      callComputeGScharr(img, grad);
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("computeGScharr", _N);
  showResult(grad, "texture scharr gradient image x");

  // compute gradient using current method
  allocate();
  callComputeG(img);
  _startGPU = chrono::high_resolution_clock::now();
  {
    for(int i=1;i<=_N;i++)
      callComputeG(img);
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("callComputeG", _N);
  showResult(_gx, "gradient image x");

  // compute gradient using in-built cv cpu functions
  Mat cpu_grad_x, cpu_grad_y;
  _startGPU = chrono::high_resolution_clock::now();
  {
    cv::Scharr(im, cpu_grad_x, CV_32F, 1, 0);
    cv::Scharr(im, cpu_grad_y, CV_32F, 0, 1);
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("Scharr CPU");

  // compute using in-built cv::gpu functions
  GpuMat grad_x, grad_y;
  _startGPU = chrono::high_resolution_clock::now();
  {
    cv::gpu::Scharr(img, grad_x, CV_32F, 1, 0);
    cv::gpu::Scharr(img, grad_y, CV_32F, 0, 1);
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("Scharr GPU");
  showResult(grad_x, "scharr gradient image x");

  return 0;
}
