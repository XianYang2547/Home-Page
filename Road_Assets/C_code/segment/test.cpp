//
// Created by xianyang on 24-9-12.
//
#include "seg_utils.h"
#include "seg_infer.h"

int main(){
    // create detector, and load engine plan
    std::string trtFile = "/home/xianyang/Desktop/Project/model_files/linuxtrt/city/n/cityseg.plan";
    YoloSeg detector(trtFile);
    // 创建 VideoCapture 对象，并打开视频文件
    cv::VideoCapture cap("/home/xianyang/Desktop/Project/test_data/test.avi");

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }
    // 创建窗口
    cv::namedWindow("Video Playback", cv::WINDOW_AUTOSIZE);

    // 用于存储每一帧
    cv::Mat frame;
    while (true) {
        // 读取一帧
        cap >> frame;

        // 如果没有读取到帧，则视频播放完毕
        if (frame.empty()) {
            break;
        }
        auto start = std::chrono::system_clock::now();

        std::vector<segDetection> res = detector.inference(frame);
        auto end = std::chrono::system_clock::now();
        int cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " cost: " << cost << " ms." << std::endl;
        YoloSeg::draw_image(frame, res);

        // 显示帧
        cv::imshow("Video Playback", frame);

        // 按下 'q' 键退出播放
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
    // 释放 VideoCapture 对象
    cap.release();

    // 关闭窗口
    cv::destroyWindow("Video Playback");
    return 0;
}