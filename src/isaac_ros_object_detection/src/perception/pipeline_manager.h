#pragma once
#include "config.h"
#include "thread_interface.h"

// 在某些 ROS2/ament 默认隐藏符号的配置下，未显式导出的自由函数
// 可能不会出现在共享库的导出符号表里，导致链接可执行文件时报 undefined
// reference。
#ifndef OBJECT_DETECTION_SYMBOL_VISIBLE
#if defined(__GNUC__) || defined(__clang__)
#define OBJECT_DETECTION_SYMBOL_VISIBLE __attribute__((visibility("default")))
#else
#define OBJECT_DETECTION_SYMBOL_VISIBLE
#endif
#endif

OBJECT_DETECTION_SYMBOL_VISIBLE
std::unique_ptr<PipelineInterface>
createPipelineManager(InputConfig input_config);
