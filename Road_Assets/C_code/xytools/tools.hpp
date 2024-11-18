//
// Created by xianyang on 24-8-22.
//

#ifndef MY_INFER_TOOLS_HPP
#define MY_INFER_TOOLS_HPP

#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include "seg_infer.h"
#include "BYTETracker.h"
#include "seg_draw.h"
#include "my_curve.hpp"

inline bool IsPathExist(const std::string &path) {
    if (access(path.c_str(), 0) == F_OK) {
        return true;
    }
    return false;
}

inline bool IsFile(const std::string &path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer{};
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string &path) {
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

std::string formatTimestamp(long long timestamp_ns) {
    // Convert nanoseconds to milliseconds
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::nanoseconds(timestamp_ns));

    // Convert milliseconds to seconds
    auto timestamp_sec = std::chrono::duration_cast<std::chrono::seconds>(timestamp_ms);

    // Convert to std::chrono::system_clock::time_point
    std::chrono::system_clock::time_point tp(timestamp_sec);

    // Convert time_point to std::tm in UTC
    std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm utc_tm = *std::gmtime(&tt);

    // Extract milliseconds from the timestamp
    int milliseconds = static_cast<int>((timestamp_ms.count() % 1000));

    // Format std::tm into desired string format
    std::ostringstream oss;
    oss << std::put_time(&utc_tm, "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << milliseconds;

    return oss.str();
}

// 得到轮廓点轮廓
std::vector<cv::Point> process_and_find_contours(const std::vector<float> &maskMatrix) {
    int h = 1080; // 高度
    int w = 1920; // 宽度

    // 1. 将 maskMatrix 转换为 cv::Mat
    cv::Mat maskMat(h, w, CV_32F, (void *) maskMatrix.data());

    // 2. 应用阈值，将值大于 0.5 的置为 1，其他置为 0
    cv::Mat binaryMat;
    cv::threshold(maskMat, binaryMat, 0.5, 1, cv::THRESH_BINARY);

    // 3. 将 binaryMat 转换为 CV_8UC1 类型，以便使用 findContours
    binaryMat.convertTo(binaryMat, CV_8UC1, 255.0);

    // 4. 找到轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryMat, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 5. 如果有轮廓，找出最大的那个
    if (!contours.empty()) {
        // 按照轮廓面积排序
        auto max_contour_it = std::max_element(contours.begin(), contours.end(),
                                               [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                                   return cv::contourArea(a) < cv::contourArea(b);
                                               });

        // 返回最大的轮廓
        return *max_contour_it;
    }

    // 如果没有轮廓，返回空的 vector
    return std::vector<cv::Point>();
}

// 从轮廓点中等距离选取一些点
std::vector<cv::Point> get_equally_spaced_points(const std::vector<cv::Point> &contour, int n) {
    std::vector<cv::Point> sampled_points;

    if (contour.empty() || n <= 1) {
        return sampled_points; // 如果轮廓为空或所需点数小于等于1，直接返回空
    }

    int total_points = contour.size();
    int step = total_points / n; // 计算选点的间隔

    for (int i = 0; i < n; ++i) {
        int index = i * step; // 根据间隔选择点
        sampled_points.push_back(contour[index]); // 添加点到结果中
    }

    return sampled_points;
}
void separateEdgesByX(const std::vector<cv::Point>& contour, std::vector<cv::Point>& leftEdge, std::vector<cv::Point>& rightEdge) {
    // 如果轮廓为空，直接返回
    if (contour.empty()) return;

    // 计算轮廓点的 x 坐标的中值
    std::vector<int> xValues;
    for (const auto& point : contour) {
        xValues.push_back(point.x);
    }

    // 按照 x 坐标排序
    std::sort(xValues.begin(), xValues.end());

    // 计算中间 x 坐标（或用最大最小值做分割）
    int medianX = xValues[xValues.size() / 2];

    // 根据中值分割点集为左边缘和右边缘
    for (const auto& point : contour) {
        if (point.x < medianX) {
            leftEdge.push_back(point);
        } else {
            rightEdge.push_back(point);
        }
    }
}
void separateEdgesByY(const std::vector<cv::Point>& contour, std::vector<cv::Point>& upperEdge, std::vector<cv::Point>& lowerEdge) {
    // 如果轮廓为空，直接返回
    if (contour.empty()) return;

    // 计算轮廓点的 y 坐标的中值
    std::vector<int> yValues;
    for (const auto& point : contour) {
        yValues.push_back(point.y);
    }

    // 按照 y 坐标排序
    std::sort(yValues.begin(), yValues.end());

    // 计算中间 y 坐标
    int medianY = yValues[yValues.size() / 2];

    // 根据中值分割点集为上下边缘
    for (const auto& point : contour) {
        if (point.y < medianY) {
            upperEdge.push_back(point);  // 上边缘
        } else {
            lowerEdge.push_back(point);  // 下边缘
        }
    }
}
int detect_img(const std::string &engine_file,
               const std::vector<std::string> &imagePathList, const std::string &output_directory) {
    YoloSeg model(engine_file);
    cv::Mat image;
    for (auto &path: imagePathList) {
        image = cv::imread(path);
        //        std::vector<Detection> res = detector.inference(image);
        std::vector<segDetection> res1 = model.inference(image);
        //        YoloDet::draw_image(image, res);

        for (auto &i: res1) {
            // 获取类名
            std::string class_Name = vClassNames[(int) i.classId];
            //  线
            if (laneClasses.find(class_Name) != laneClasses.end()) {
                std::vector<cv::Point> contours = process_and_find_contours(i.maskMatrix);
                std::vector<cv::Point> leftEdge;
                std::vector<cv::Point> rightEdge;
                separateEdgesByX(contours,leftEdge,rightEdge);

                for (const auto& point : leftEdge) {
                    cv::circle(image, point, 5, cv::Scalar(255, 0, 0), -1); // 红色圆点，半径为5
                }
                for (const auto& point : rightEdge) {
                    cv::circle(image, point, 5, cv::Scalar(0, 255, 0), -1); // 红色圆点，半径为5
                }


            }
            // 目标
            if (bboxClasses.find(class_Name) != bboxClasses.end()) {
                float x1 = round(i.bbox[0]);
                float y1 = round(i.bbox[1]);
                float x2 = round(i.bbox[2]);
                float y2 = round(i.bbox[3]);
                float center_x = (x1 + x2) / 2;
                float center_y = (y1 + y2) / 2;
                //写入3d点
            }
            // 其他的
            if (other.find(class_Name) != other.end()) {
                std::vector<cv::Point> contours = process_and_find_contours(i.maskMatrix);
                std::vector<cv::Point> sampled_points = get_equally_spaced_points(contours,5);
                //写入3d点
            }

        }
        YoloSeg::draw_image(image, res1);
        size_t last_slash_idx = path.find_last_of("\\/");
        size_t last_dot_idx = path.find_last_of(".");
        std::string filename;
        if (last_slash_idx == std::string::npos) {
            last_slash_idx = -1;
        }
        if (last_dot_idx == std::string::npos || last_dot_idx < last_slash_idx) {
            last_dot_idx = path.length();
        }
        filename = path.substr(last_slash_idx + 1, last_dot_idx - last_slash_idx - 1);
        std::string new_path = output_directory + filename + "_result.jpg";
        cv::imwrite(new_path, image);
    }
}


bool isTrackingsegClass(int class_id, const std::vector<int> &tracksegClasses) {
    for (auto &c: tracksegClasses) {
        if (class_id == c) return true;
    }
    return false;
}


int detect_mp4(const std::string &engine_file, char *videoPath, const std::vector<int> &tracksegClasses,
               const std::string &output_mp4) {
    std::string inputVideoPath = std::string(videoPath);
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) return 0;
    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;
    cv::VideoWriter writer(output_mp4, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    // 加载引擎文件
    YoloSeg model(engine_file);
    // ByteTrack tracker
    BYTETracker tracker(fps, 30);

    cv::Mat img;
    int num_frames = 0;
    int total_ms = 0;
    cv::namedWindow("res", cv::WINDOW_NORMAL);
    while (true) {
        if (!cap.read(img)) break;
        num_frames++;
        if (img.empty()) break;
        auto start = std::chrono::system_clock::now();
        // inference
        std::vector<segDetection> res1 = model.inference(img);
        // output format to bytetrack input format, and filter bbox by class id
        std::vector<Object> objects;
        for (auto &j: res1) {
            float *bbox = j.bbox;
            float conf = j.conf;
            int classId = j.classId;

            if (isTrackingsegClass(classId, tracksegClasses)) {
                cv::Rect_<float> rect(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]));
                Object obj{rect, classId, conf};
                objects.push_back(obj);
            }
        }
        // track update
        std::vector<STrack> output_stracks = tracker.update(objects);

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for (auto &output_strack: output_stracks) {
            std::vector<float> tlwh = output_strack.tlwh;
            if (tlwh[2] * tlwh[3] > 20) {
                cv::Scalar s = tracker.get_color(output_strack.track_id);
                cv::putText(img, cv::format("%d", output_strack.track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                            0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                // cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }
        // seg draw
        for (auto &i: res1) {
            draw_mask(img, i.maskMatrix.data());
        }

        cv::putText(img, cv::format("frame: %d fps: %d num: %ld", num_frames, num_frames * 1000000 / total_ms,
                                    output_stracks.size()),
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        writer.write(img);

        cv::imshow("res", img);
        cv::resizeWindow("res", 800, 600);
        int cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " cost: " << cost << " ms" << std::endl;
        char c = waitKey(1);
        if (c > 0) break;
    }

    cap.release();
    writer.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;
}

#endif //MY_INFER_TOOLS_HPP
