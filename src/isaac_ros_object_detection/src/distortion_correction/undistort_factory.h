// 定义工厂函数创建实例
#pragma once

#include "undistorter_interface.h"
#include "undistort_cpu.h"
#include "undistort_gpu.h"
#include "perception/config.h"

// 工厂函数：根据配置创建 CPU 或 GPU 去畸变器
std::unique_ptr<UndistorterInterface> createUndistorter(
    UndistortConfig &undistort_config,
    InputConfig &input_config);