//
// Created by xianyang on 24-8-21.
//

#ifndef MY_INFER_SEG_INFER_H
#define MY_INFER_SEG_INFER_H

#include <opencv2/opencv.hpp>
#include "seg_public.h"
#include "seg_types.h"
#include "seg_config.h"

using namespace nvinfer1;


class YoloSeg {
public:
    YoloSeg(const std::string trtFile);

    ~YoloSeg();

    std::vector<segDetection> inference(cv::Mat &img);

    static void draw_image(cv::Mat &img, std::vector<segDetection> &inferResult, bool drawBbox = true);

private:
    void get_engine();

    static void process_mask(
            float *protoDevice, Dims32 protoOutDims, std::vector<segDetection> &vDetections,
            int kInputH, int kInputW, cv::Mat &img, cudaStream_t stream
    );

private:
    Logger gLogger;
    std::string trtFile_;

    ICudaEngine *engine;
    IRuntime *runtime;
    IExecutionContext *context;

    cudaStream_t stream;

    float *outputData;
    std::vector<void *> vBufferD;
    float *transposeDevide;
    float *decodeDevice;

    int OUTPUT_CANDIDATES;  // 8400: 80 * 80 + 40 * 40 + 20 * 20
    Dims32 protoOutDims;  // proto shape [1 32 160 160]
};

#endif //MY_INFER_SEG_INFER_H
