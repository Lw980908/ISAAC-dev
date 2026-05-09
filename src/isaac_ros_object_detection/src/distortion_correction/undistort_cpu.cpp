#include "undistort_cpu.h"
#include <logger.h>

UndistorterCPU::UndistorterCPU(UndistortConfig &undistort_config, InputConfig &input_config)
{
    int h = input_config.input_sizes[input_config.idx].height;
    int w = input_config.input_sizes[input_config.idx].width;

    if (h <= 0 || w <= 0)
    {
        throw std::runtime_error("Invalid image dimensions for undistortion");
    }

    // 在构造函数中初始化 roi_
    // roi_ = cv::Rect(0, 0, w, h);

    if (undistort_config.camera_type_groups[input_config.idx] == CameraType::Fish_eye_camera)
    {
        sample::gLogInfo << "Undistort on CPU: Using Fisheye camera model." << std::endl;
        const auto &params = undistort_config.Fisheye_param_groups[input_config.idx];

        // 初始化相机内参矩阵（手动调整焦距）
        // 手动将 fx 和 fy 设置为更大的值，以增加输出图像的有效区域
        // cx 和 cy 设置为图像的中心点，以使图像中心对齐（图像中心的移动 -> 输出图像区域的变化）
        cv::Mat camera_matrix_ = (cv::Mat_<double>(3, 3) << params.fx * 1.2, 0, params.cx, 0, params.fy * 1.2, params.cy, 0, 0, 1);

        // 初始化畸变系数
        cv::Mat dist_coeffs_ = (cv::Mat_<double>(4, 1) << params.k1, params.k2, params.k3, params.k4);

        // 计算新的相机矩阵和ROI，设置缩放因子为1
        // getOptimalNewCameraMatrix函数的第四个参数是 alpha，控制了去畸变时是否保留图像的全尺寸。
        // alpha 值在 [0, 1] 之间，alpha = 0 时会裁剪图像，alpha = 1 时会保留全尺寸图像。
        cv ::Mat new_camera_matrix_ = cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, cv::Size(w, h), 0, cv::Size(w, h), &roi_);
        // cv ::Mat new_camera_matrix_ = cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, cv::Size(w, h), 0, cv::Size(w, h));

        // 创建去畸变映射表
        cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat::eye(3, 3, CV_32F), new_camera_matrix_, cv::Size(w, h), CV_32F, map1_, map2_);
    }
    else if (undistort_config.camera_type_groups[input_config.idx] == CameraType::Pin_hole_camera)
    {
        sample::gLogInfo << "Undistort on CPU: Using pinhole camera model." << std::endl;
        const auto &params = undistort_config.Pin_hole_param_groups[input_config.idx];

        // 初始化相机内参矩阵（手动调整焦距）
        cv::Mat camera_matrix_ = (cv::Mat_<double>(3, 3) << params.fx * 1.5, 0, params.cx, 0, params.fy, params.cy, 0, 0, 1);

        // 初始化畸变系数
        cv::Mat dist_coeffs_ = (cv::Mat_<double>(8, 1) << params.k1, params.k2, params.p1, params.p2,
                                params.k3, params.k4, params.k5, params.k6);

        // 计算新的相机矩阵和ROI，设置缩放因子为1
        cv ::Mat new_camera_matrix_ = cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, cv::Size(w, h), 0, cv::Size(w, h), &roi_);
        // cv ::Mat new_camera_matrix_ = cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, cv::Size(w, h), 0, cv::Size(w, h));

        // 创建去畸变映射表
        cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(), new_camera_matrix_, cv::Size(w, h), CV_32FC1, map1_, map2_);
    }
    else
    {
        throw std::runtime_error("Unsupported camera type for CPU undistortion!");
    }

    sample::gLogInfo << "Initialize CPU undistortion successfully." << std::endl;
}

// 执行图像去畸变
cv::Mat UndistorterCPU::undistort_image(const cv::Mat &image)
{
    if (map1_.empty() || map2_.empty())
    {
        throw std::runtime_error("UndistorterCPU未初始化");
    }

    cv::Mat undistorted;
    cv::remap(image, undistorted, map1_, map2_, cv::INTER_LINEAR); // cv::INTER_CUBIC
    return undistorted(roi_);
}
