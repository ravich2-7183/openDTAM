#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <chrono>
#include <string>
#include <iostream>
#include <stdio.h>
#include "computeG.cuh"
#include "update_qd.cuh"

using namespace cv;
using namespace cv::gpu;
using namespace std;

const int _N = 100;

int _rows, _cols;

GpuMat _g1, _gx, _gy;
GpuMat _qx, _qy;

#define FLATALLOC(a,m) a.create(1,m.rows*m.cols, CV_32FC1); a = a.reshape(0,m.rows)
void allocate() {
  _g1.create(1,_rows*_cols, CV_32FC1);
  _g1=_g1.reshape(0,_rows);

  FLATALLOC(_gx,_g1);
  FLATALLOC(_gy,_g1);

  FLATALLOC(_qx, _g1);
  FLATALLOC(_qy, _g1);
}

void callComputeG(GpuMat& im){
  float* pp  = (float*) im.data;
  float* g1p = (float*)_g1.data;
  float* gxp = (float*)_gx.data;
  float* gyp = (float*)_gy.data;
  computeGCaller(pp,  g1p,  gxp,  gyp, _rows, _cols);
}

float _sigma_d, _sigma_q;
void computeSigmas(float epsilon, float theta, float L=4)
{
  float mu = 2.0*sqrt(epsilon/theta)/L;
  _sigma_d = mu*theta/2.0;
  _sigma_q = mu/(2.0*epsilon);
}

auto _startGPU = chrono::high_resolution_clock::now();
auto _endGPU   = chrono::high_resolution_clock::now();
void printTime(string codelabel, float N = 1.0f) {
  cout << codelabel
       << " : ----------------"
       <<  (float)(chrono::duration_cast<chrono::microseconds>(_endGPU - _startGPU).count())/N
       << " usec"
       << endl;
}

Mat _disp_grad;
void showResult(GpuMat& result, string windowname) {
  result.download(_disp_grad);
  cout << "absSum = " << sum(abs(_disp_grad)) << endl;
  namedWindow(windowname, WINDOW_AUTOSIZE);
  imshow(windowname, _disp_grad);
}

void showResultCPU(Mat& result, string windowname) {
  cout << "absSum = " << sum(abs(result)) << endl;
  namedWindow(windowname, WINDOW_AUTOSIZE);
  imshow(windowname, result);
}

int main(){
  // load image
  char filename[500];
  Mat im;
  sprintf(filename, "../../../../Trajectory_30_seconds/scene_%03d.png", 20);
  printf("Opening: %s \n", filename);
  imread(filename, -1).convertTo(im, CV_32FC3, 1.0/65535.0);
  cvtColor(im, im, CV_RGB2GRAY);

  // namedWindow("input image", WINDOW_AUTOSIZE );
  // imshow("input image", im );
  // waitKey(0);

  _rows=im.rows;
  _cols=im.cols;

  // add noise to image
  float mu = 0.0, sigma=0.1;
  Mat noise = Mat(im.size(), CV_32F);
  randn(noise, mu, sigma);
  noise = im + noise;

  namedWindow("input image with noise", WINDOW_AUTOSIZE );
  imshow("input image with noise", noise);
  waitKey(0);

  GpuMat img(im);
  GpuMat imgn(noise);

  float theta=1.0f, epsilon=0.1f;
  computeSigmas(epsilon, theta);

  GpuMat g(_rows, _cols, CV_32FC1); // image gradients computed on the noisy image, imgn
  computeGScharrCaller((float*)imgn.data, (float*)g.data, imgn.cols, imgn.rows, imgn.step);
  GpuMat a, d;
  GpuMat q(2*_rows, _cols, CV_32FC1, 0.0f);

  float width=img.cols, height=img.rows, pitch=img.step;

  // update qd using only cpu
  Mat gc(_rows, _cols, CV_32FC1); // image gradients computed on the noisy image, imgn
  g.download(gc);
  Mat ac, dc;
  Mat qc(2*_rows, _cols, CV_32FC1, 0.0f);

  ac = noise.clone(); dc = noise.clone();

  _startGPU = chrono::high_resolution_clock::now();
  {
    for(int i=1;i<=_N;i++) {
      update_qdCPU((float *)gc.data, (float *)ac.data,  // const input
                   (float *)qc.data, (float *)dc.data,  // input q, d
                   width, height, // dimensions
                   _sigma_q, _sigma_d, epsilon, theta // parameters
                   );
      // showResult(d, "denoised with update_qdCaller");
      // waitKey(30);
    }
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("update_q_dCPU, gold standard for correctness", _N);
  showResultCPU(dc, "denoised with update_q_dCPU");
  waitKey(0);

  // denoise image using update_q_d with textures
  a = imgn.clone(); d = imgn.clone();
  update_q_d_BindTextures((float*)q.data, (float*)d.data, imgn.cols, imgn.rows, imgn.step);
  update_q_dCaller((float*)g.data,  (float*)a.data,  // input
                   (float*)q.data,  (float*)d.data,  // input  q, d
                   width, height, pitch, // dimensions
                   _sigma_q, _sigma_d, epsilon, theta // parameters
                   );
  _startGPU = chrono::high_resolution_clock::now();
  {
    for(int i=1;i<=_N;i++) {
      update_q_dCaller((float*)g.data,  (float*)a.data,  // input
                       (float*)q.data,  (float*)d.data,  // input  q, d
                       width, height, pitch, // dimensions
                       _sigma_q, _sigma_d, epsilon, theta // parameters
                       );
      // showResult(d, "denoised with update_q_dCaller");
      // waitKey(30);
    }
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("update_q_dCaller uses texture memory, correctness off", _N);
  showResult(d, "denoised with update_q_dCaller");
  waitKey(0);

  // denoise image using update_qd combined no textures
  a = imgn.clone(); d = imgn.clone();
  q = 0.0f;
  update_qdCaller((float*)g.data,  (float*)a.data,  // input
                  (float*)q.data,  (float*)d.data,  // input  q, d
                  width, height, // dimensions
                  _sigma_q, _sigma_d, epsilon, theta // parameters
                  );
  _startGPU = chrono::high_resolution_clock::now();
  {
    for(int i=1;i<=_N;i++) {
      update_qdCaller((float*)g.data,  (float*)a.data,  // input
                      (float*)q.data,  (float*)d.data,  // input  q, d
                      width, height, // dimensions
                      _sigma_q, _sigma_d, epsilon, theta // parameters
                      );
      // showResult(d, "denoised with update_q_dCaller");
      // waitKey(30);
    }
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("update_qdCaller, correctness NOT gauranteed", _N);
  showResult(d, "denoised with update_qdCaller");
  waitKey(0);

  // denoise image using original method, updateQDCaller
  a = imgn.clone(); d = imgn.clone();
  allocate();
  _qx = 0.0f; _qy = 0.0f;
  callComputeG(imgn); // image gradients computed on the noisy image, imgn
  float* dpt   = (float*)d.data;
  float* apt   = (float*)a.data;
  float* gxpt  = (float*)_gx.data;
  float* gypt  = (float*)_gy.data;
  float* gqxpt = (float*)_qx.data;
  float* gqypt = (float*)_qy.data;
  updateQDCaller(gqxpt, gqypt, dpt, apt,
                 gxpt, gypt, _cols, _rows,
                 _sigma_q, _sigma_d, epsilon, theta);
  _startGPU = chrono::high_resolution_clock::now();
  {
    for(int i=1;i<=_N;i++) {
      updateQDCaller(gqxpt, gqypt, dpt, apt,
                     gxpt, gypt, _cols, _rows,
                     _sigma_q, _sigma_d, epsilon, theta);
      // showResult(d, "denoised with updateQD original");
      // waitKey(30);
    }
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("updateQDCaller", _N);
  showResult(d, "denoised with updateQD original");
  waitKey(0);

  // denoise image using update_qd_new
  a = imgn.clone(); d = imgn.clone();
  q = 0.0f;
  update_q_d_NoTexCaller((float*)g.data, (float*)a.data,  // input
                         (float*)q.data, (float*)d.data,  // input  q, d
                         width, height, // dimensions
                         _sigma_q, _sigma_d, epsilon, theta // parameters
                         );
  _startGPU = chrono::high_resolution_clock::now();
  {
    for(int i=1;i<=_N;i++) {
      update_q_d_NoTexCaller((float*)g.data, (float*)a.data,  // input
                             (float*)q.data, (float*)d.data,  // input  q, d
                             width, height, // dimensions
                             _sigma_q, _sigma_d, epsilon, theta // parameters
                             );
      // showResult(d, "denoised with update_qdCaller");
      // waitKey(30);
    }
  }
  _endGPU = chrono::high_resolution_clock::now();
  printTime("update_q_dNoTexCaller", _N);
  showResult(d, "denoised with update_q_dNoTexdCaller");
  waitKey(0);
  
  return 0;
}
