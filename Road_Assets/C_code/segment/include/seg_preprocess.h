//
// Created by xianyang on 24-8-21.
//

#ifndef MY_INFER_SEG_PREPROCESS_H
#define MY_INFER_SEG_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void preprocess(const cv::Mat &srcImg, float *dstDevData, const int dstHeight, const int dstWidth, cudaStream_t stream);
/*
srcImg:     source image for inference
dstDevData: data after preprocess (resize / bgr to rgb / hwc to chw / normalize)
dstHeight:  CNN input height
dstWidth:   CNN input width
*/

#endif //MY_INFER_SEG_PREPROCESS_H
