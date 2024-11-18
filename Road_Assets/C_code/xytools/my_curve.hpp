//
// Created by xianyang on 24-6-24.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "eigen3/Eigen/Dense"


void groupPointsByY(const std::vector<cv::Point>& points, std::map<int, std::vector<int>>& groupedPoints) {
    for (const auto& point : points) {
        groupedPoints[point.y].push_back(point.x);
    }
}
void calculateCenterPoints(const std::map<int, std::vector<int>>& groupedPoints, std::vector<cv::Point>& centerPoints) {
    for (const auto& group : groupedPoints) {
        int sumX = 0;
        for (int x : group.second) {
            sumX += x;
        }
        int centerX = sumX / group.second.size();
        centerPoints.emplace_back(centerX, group.first);
    }
}

void fitPolynomial(const std::vector<cv::Point>& points, std::vector<cv::Point>& fittedPoints, int degree) {
    // Extract x and y coordinates
    int n = points.size();
    cv::Mat X(n, degree + 1, CV_64F);
    cv::Mat Y(n, 1, CV_64F);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < degree + 1; ++j) {
            X.at<double>(i, j) = pow(points[i].x, j);
        }
        Y.at<double>(i, 0) = points[i].y;
    }

    // Solve for the polynomial coefficients
    cv::Mat coeffs = (X.t() * X).inv() * X.t() * Y;

    // Generate fitted points
    for (int x = 0; x < 500; ++x) { // 800 is the width of the image
        double y = 0;
        for (int j = 0; j < degree + 1; ++j) {
            y += coeffs.at<double>(j, 0) * pow(x, j);
        }
        fittedPoints.emplace_back(x, static_cast<int>(y));
    }
}

void downsamplePoints(const std::vector<cv::Point>& inputPoints, std::vector<cv::Point>& outputPoints, int step) {
    for (size_t i = 0; i < inputPoints.size(); i += step) {
        int sumX = 0, sumY = 0;
        int count = 0;
        for (size_t j = i; j < i + step && j < inputPoints.size(); ++j) {
            sumX += inputPoints[j].x;
            sumY += inputPoints[j].y;
            ++count;
        }
        outputPoints.emplace_back(sumX / count, sumY / count);
    }
}
#ifndef MY_INFER_MY_CURVE_HPP
#define MY_INFER_MY_CURVE_HPP

#endif //MY_INFER_MY_CURVE_HPP
