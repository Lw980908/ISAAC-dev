#include "undistort_gpu.h"

// 只有在 OpenCV CUDA 可用时才编译 GPU 实现
#if OPENCV_CUDA_AVAILABLE

#include <logger.h>
// 性能监控宏
#define ENABLE_PERF_MONITOR 1

#if ENABLE_PERF_MONITOR
#define PERF_START() auto perf_start = std::chrono::high_resolution_clock::now()
#define PERF_END(operation)                                                                                    \
    auto perf_end = std::chrono::high_resolution_clock::now();                                                 \
    auto perf_duration = std::chrono::duration_cast<std::chrono::microseconds>(perf_end - perf_start).count(); \
    if (++total_frames_ % 100 == 0)                                                                            \
    {                                                                                                          \
        sample::gLogInfo << "GPU " << operation << " time: " << perf_duration << "us" << std::endl;            \
    }                                                                                                          \
    total_processing_time_us_ += perf_duration
#else
#define PERF_START()
#define PERF_END(operation)
#endif

UndistorterGPU::UndistorterGPU(UndistortConfig &undistort_config, InputConfig &input_config)
{
    try
    {
        initialize_maps(undistort_config, input_config);
        initialized_ = true;
        sample::gLogInfo << "GPU undistorter initialized successfully." << std::endl;
    }
    catch (const std::exception &e)
    {
        sample::gLogError << "Failed to initialize GPU undistorter: " << e.what() << std::endl;
        throw;
    }
}

UndistorterGPU::~UndistorterGPU()
{
    // 等待所有异步操作完成
    if (stream_.queryIfComplete() == false)
    {
        stream_.waitForCompletion();
    }

    // 输出性能统计
    if (total_frames_ > 0)
    {
        sample::gLogInfo << "GPU undistorter processed " << total_frames_
                         << " frames, average time: "
                         << (total_processing_time_us_ / total_frames_) << "us per frame" << std::endl;
    }
}

void UndistorterGPU::initialize_maps(UndistortConfig &undistort_config, InputConfig &input_config)
{
    int h = input_config.input_sizes[input_config.idx].height;
    int w = input_config.input_sizes[input_config.idx].width;

    if (h <= 0 || w <= 0)
    {
        throw std::runtime_error("Invalid image dimensions for undistortion!");
    }

    if (undistort_config.camera_type_groups[input_config.idx] == CameraType::Fish_eye_camera)
    {
        sample::gLogInfo << "Undistort on GPU: Using Fisheye camera model." << std::endl;
        const auto &params = undistort_config.Fisheye_param_groups[input_config.idx];

        // 初始化相机内参矩阵
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << params.fx, 0, params.cx,
                                 0, params.fy, params.cy,
                                 0, 0, 1);

        // 初始化畸变系数
        cv::Mat dist_coeffs = (cv::Mat_<double>(4, 1) << params.k1, params.k2, params.k3, params.k4);

        // 计算新相机矩阵和ROI
        cv::Rect roi;
        cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, cv::Size(w, h), 0, cv::Size(w, h), &roi);
        roi_ = roi;

        // 生成映射表
        cv::Mat map1, map2;
        cv::fisheye::initUndistortRectifyMap(
            camera_matrix, dist_coeffs, cv::Mat::eye(3, 3, CV_32F),
            new_camera_matrix, cv::Size(w, h), CV_32F, map1, map2);

        // 将映射表上传到GPU
        d_map1_.upload(map1, stream_);
        d_map2_.upload(map2, stream_);
    }
    else if (undistort_config.camera_type_groups[input_config.idx] == CameraType::Pin_hole_camera)
    {
        sample::gLogInfo << "Undistort on GPU: Using pinhole camera model." << std::endl;
        const auto &params = undistort_config.Pin_hole_param_groups[input_config.idx];

        // 初始化相机内参矩阵
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << params.fx * 1.5, 0, params.cx,
                                 0, params.fy, params.cy,
                                 0, 0, 1);

        // 初始化畸变系数
        cv::Mat dist_coeffs = (cv::Mat_<double>(8, 1) << params.k1, params.k2, params.p1, params.p2,
                               params.k3, params.k4, params.k5, params.k6);

        // 计算新相机矩阵和ROI
        cv::Rect roi;
        cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, cv::Size(w, h), 0, cv::Size(w, h), &roi, false);
        roi_ = roi;

        // 生成映射表
        cv::Mat map1, map2;
        cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(),
                                    new_camera_matrix, cv::Size(w, h), CV_32FC1, map1, map2);

        // 将映射表上传到GPU
        d_map1_.upload(map1, stream_);
        d_map2_.upload(map2, stream_);
    }
    else
    {
        throw std::runtime_error("Unsupported camera type for GPU undistortion!");
    }

    // 等待映射表上传完成
    stream_.waitForCompletion();
}

cv::Mat UndistorterGPU::undistort_image(const cv::Mat &image)
{
    return undistort_image_optimized(image);
}

cv::Mat UndistorterGPU::undistort_image_optimized(const cv::Mat &image)
{
    if (!initialized_)
    {
        throw std::runtime_error("UndistorterGPU not initialized");
    }

    PERF_START();

    std::lock_guard<std::mutex> lock(buffer_mutex_);

    // 检查并初始化重用缓冲区
    if (!buffer_initialized_ ||
        d_input_buffer_.size() != image.size() ||
        d_input_buffer_.type() != image.type())
    {

        d_input_buffer_.create(image.rows, image.cols, image.type());
        d_output_buffer_.create(image.rows, image.cols, image.type());

        if (!roi_.empty())
        {
            d_roi_buffer_.create(roi_.height, roi_.width, image.type());
        }

        buffer_initialized_ = true;
        sample::gLogInfo << "GPU buffers reinitialized for size: "
                         << image.cols << "x" << image.rows << std::endl;
    }

    try
    {
        // 异步上传到GPU
        d_input_buffer_.upload(image, stream_);

        // 异步remap操作
        cv::cuda::remap(d_input_buffer_, d_output_buffer_, d_map1_, d_map2_,
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), stream_);

        cv::Mat result;

        if (!roi_.empty() && roi_.area() > 0)
        {
            // 异步提取ROI区域
            cv::cuda::GpuMat d_roi = d_output_buffer_(roi_);

            // 异步下载ROI区域
            d_roi.download(result, stream_);
        }
        else
        {
            // 异步下载完整图像
            d_output_buffer_.download(result, stream_);
        }

        // 等待所有异步操作完成
        stream_.waitForCompletion();

        PERF_END("undistort_optimized");
        return result;
    }
    catch (const cv::Exception &e)
    {
        sample::gLogError << "GPU undistort error: " << e.what() << std::endl;
        // 降级处理：返回原图
        return image.clone();
    }
    catch (const std::exception &e)
    {
        sample::gLogError << "GPU undistort exception: " << e.what() << std::endl;
        return image.clone();
    }
}

std::future<cv::Mat> UndistorterGPU::undistort_image_async(const cv::Mat &image)
{
    return std::async(std::launch::async, [this, image]()
                      { return this->undistort_image_optimized(image); });
}

cv::Mat UndistorterGPU::undistort_image_internal(const cv::Mat &image, bool async_mode)
{
    // 基本实现，与优化版本类似但更简单
    if (!initialized_)
    {
        throw std::runtime_error("UndistorterGPU not initialized");
    }

    PERF_START();

    // 上传输入图像到GPU
    cv::cuda::GpuMat d_image;
    d_image.upload(image, stream_);

    // 执行GPU加速的remap操作
    cv::cuda::GpuMat d_undistorted;
    cv::cuda::remap(d_image, d_undistorted, d_map1_, d_map2_,
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), stream_);

    cv::Mat result;

    if (!roi_.empty() && roi_.area() > 0)
    {
        // 提取ROI区域
        cv::cuda::GpuMat d_roi = d_undistorted(roi_);
        d_roi.download(result, stream_);
    }
    else
    {
        d_undistorted.download(result, stream_);
    }

    // 等待CUDA流完成
    stream_.waitForCompletion();

    PERF_END("undistort_basic");
    return result;
}

#endif // OPENCV_CUDA_AVAILABLE