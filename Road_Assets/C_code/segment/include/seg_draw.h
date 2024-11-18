//
// Created by xianyang on 24-8-21.
//

#ifndef MY_INFER_SEG_DRAW_H
#define MY_INFER_SEG_DRAW_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void draw_mask(cv::Mat &img, float *mask);

#endif //MY_INFER_SEG_DRAW_H
