#ifndef UNDISTORT_CPU_H
#define UNDISTORT_CPU_H

#include "perception/config.h"
#include "undistorter_interface.h"

class UndistorterCPU : public UndistorterInterface
{
public:
    UndistorterCPU(UndistortConfig &undistort_config, InputConfig &input_config);

    // 执行图像去畸变
    cv::Mat undistort_image(const cv::Mat &image) override;

private:
    mutable cv::Rect roi_;
    mutable cv::Mat map1_, map2_;
};

#endif // UNDISTORT_CPU_H