#ifndef UNDISTORT_GPU_H
#define UNDISTORT_GPU_H

#include "perception/config.h"
#include "undistorter_interface.h"

// 检查 OpenCV CUDA 模块是否可用
#ifdef HAVE_OPENCV_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#define OPENCV_CUDA_AVAILABLE 1
#else
// 尝试检测 OpenCV 版本宏
#include <opencv2/core/version.hpp>
#if defined(CV_VERSION_MAJOR) && CV_VERSION_MAJOR >= 4
    // OpenCV 4.x - 尝试包含 CUDA 头文件
    #if __has_include(<opencv2/cudaarithm.hpp>)
        #include <opencv2/cudaarithm.hpp>
        #include <opencv2/cudawarping.hpp>
        #define OPENCV_CUDA_AVAILABLE 1
    #else
        #define OPENCV_CUDA_AVAILABLE 0
    #endif
#else
    #define OPENCV_CUDA_AVAILABLE 0
#endif
#endif

#include <atomic>
#include <future>

#if OPENCV_CUDA_AVAILABLE

class UndistorterGPU : public UndistorterInterface
{
public:
    UndistorterGPU(UndistortConfig &undistort_config, InputConfig &input_config);
    ~UndistorterGPU() override;

    // 执行GPU加速的图像去畸变
    cv::Mat undistort_image(const cv::Mat &image) override;

    // 异步版本 - 立即返回，通过future获取结果
    std::future<cv::Mat> undistort_image_async(const cv::Mat &image);

    // 性能优化版本 - 重用GPU内存
    cv::Mat undistort_image_optimized(const cv::Mat &image);

    // 检查是否已初始化
    bool is_initialized() const { return initialized_; }

private:
    // 内部实现方法
    cv::Mat undistort_image_internal(const cv::Mat &image, bool async_mode = false);
    void initialize_maps(UndistortConfig &undistort_config, InputConfig &input_config);

    // GPU资源
    cv::cuda::GpuMat d_map1_;
    cv::cuda::GpuMat d_map2_;
    mutable cv::cuda::Stream stream_; // CUDA流用于异步操作

    // 重用缓冲区
    mutable cv::cuda::GpuMat d_input_buffer_;
    mutable cv::cuda::GpuMat d_output_buffer_;
    mutable cv::cuda::GpuMat d_roi_buffer_;

    // ROI区域
    cv::Rect roi_;

    // 状态标志
    std::atomic<bool> initialized_{false};
    std::atomic<bool> buffer_initialized_{false};

    // 性能监控
    mutable std::atomic<uint64_t> total_frames_{0};
    mutable std::atomic<uint64_t> total_processing_time_us_{0};

    // 线程安全
    mutable std::mutex buffer_mutex_;
};

#else // OPENCV_CUDA_AVAILABLE == 0

// OpenCV CUDA 不可用时的占位类（使用 CPU 实现回退）
#include "undistort_cpu.h"
#include <stdexcept>

class UndistorterGPU : public UndistorterInterface
{
public:
    UndistorterGPU(UndistortConfig &undistort_config, InputConfig &input_config)
        : cpu_fallback_(undistort_config, input_config)
    {
        // OpenCV CUDA 模块不可用，回退到 CPU 实现
        std::cerr << "[Warning] OpenCV CUDA module not available, falling back to CPU undistortion." << std::endl;
    }
    
    ~UndistorterGPU() override = default;

    cv::Mat undistort_image(const cv::Mat &image) override
    {
        return cpu_fallback_.undistort_image(image);
    }

    std::future<cv::Mat> undistort_image_async(const cv::Mat &image)
    {
        return std::async(std::launch::async, [this, image]() {
            return this->undistort_image(image);
        });
    }

    cv::Mat undistort_image_optimized(const cv::Mat &image)
    {
        return cpu_fallback_.undistort_image(image);
    }

    bool is_initialized() const { return true; }

private:
    UndistorterCPU cpu_fallback_;
};

#endif // OPENCV_CUDA_AVAILABLE

#endif // UNDISTORT_GPU_H