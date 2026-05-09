#include "undistort_factory.h"
#include <logger.h>

std::unique_ptr<UndistorterInterface> createUndistorter(
    UndistortConfig &undistort_config,
    InputConfig &input_config)
{
    if (undistort_config.need_undistort_)
    {
        // 根据配置决定使用 CPU 还是 GPU 实现
        if (undistort_config.use_gpu_undistort_)
        {
            return std::make_unique<UndistorterGPU>(undistort_config, input_config);
        }
        else
        {
            return std::make_unique<UndistorterCPU>(undistort_config, input_config);
        }
    }
    else
    {
        sample::gLogInfo << "No need to undistort on CPU or GPU." << std::endl;
        return nullptr;
    }
}