#pragma once

#include <cuda_runtime.h>

#include "config.h"
#include "yolo_model.h"

namespace nv12_bucket_second_detection {
struct Params {
  float bucket_min_confidence = 0.25f;
};

void Run(YOLOModel *second_yolo, const void *nv12_gpu, int width, int height,
         int y_stride, int uv_stride, cudaEvent_t ready_event,
         const std::vector<Box> &first_boxes,
         const std::vector<std::string> &first_class_names,
         ToothDetectionConfig &tooth_detection_config,
         std::vector<Box> &out_boxes, const Params &params = Params{});
} // namespace nv12_bucket_second_detection
