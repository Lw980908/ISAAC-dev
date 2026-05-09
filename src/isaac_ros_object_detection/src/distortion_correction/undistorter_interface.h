// 定义统一接口类
#pragma once
#include <opencv2/opencv.hpp>

class UndistorterInterface
{
public:
    virtual ~UndistorterInterface() = default;

    // 统一的去畸变接口
    virtual cv::Mat undistort_image(const cv::Mat &image) = 0;
};