#include "nv12_bucket_second_detection.h"

#include <algorithm>

namespace nv12_bucket_second_detection {
namespace {
int FindClassIndex(const std::vector<std::string> &names,
                   const std::string &target) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == target) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int ClampInt(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }
} // namespace

void Run(YOLOModel *second_yolo, const void *nv12_gpu, int width, int height,
         int y_stride, int uv_stride, cudaEvent_t ready_event,
         const std::vector<Box> &first_boxes,
         const std::vector<std::string> &first_class_names,
         ToothDetectionConfig &tooth_detection_config,
         std::vector<Box> &out_boxes, const Params &params) {
  out_boxes.clear();

  const int reset_idx = FindClassIndex(first_class_names, "reset");
  const int bucket_idx = FindClassIndex(first_class_names, "bucket");

  Box best_bucket;
  bool has_bucket = false;

  for (const auto &b : first_boxes) {
    const bool is_bucket = (reset_idx >= 0 && b.label == reset_idx) ||
                           (bucket_idx >= 0 && b.label == bucket_idx);
    if (is_bucket && b.confidence >= params.bucket_min_confidence) {
      if (!has_bucket || b.confidence > best_bucket.confidence) {
        best_bucket = b;
        has_bucket = true;
      }
    }
  }
  if (has_bucket) {
    out_boxes.push_back(best_bucket);
  }
  if (!has_bucket || !second_yolo || !nv12_gpu) {
    return;
  }

  int x = static_cast<int>(best_bucket.left);
  int y = static_cast<int>(best_bucket.top);
  int w = static_cast<int>(best_bucket.right - best_bucket.left);
  int h = static_cast<int>(best_bucket.bottom - best_bucket.top);

  x = ClampInt(x, 0, width - 1);
  y = ClampInt(y, 0, height - 1);
  w = ClampInt(w, 1, width - x);
  h = ClampInt(h, 1, height - y);

  auto &y2 = second_yolo->getYolo();
  y2.copyFromGxfNv12GpuRoiResize(nv12_gpu, width, height, y_stride, uv_stride,
                                 x, y, w, h, ready_event, 0);
  y2.preprocess();
  y2.infer();
  y2.postprocess(1);
  auto objectss = y2.getObjectss();
  y2.reset();

  if (objectss.empty() || objectss[0].empty()) {
    return;
  }

  const auto &names2 = second_yolo->getParam().class_names;
  const int in_w = std::max(1, second_yolo->getParam().src_w);
  const int in_h = std::max(1, second_yolo->getParam().src_h);
  const float sx = static_cast<float>(w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(h) / static_cast<float>(in_h);

  for (auto box : objectss[0]) {
    if (box.label < 0 || static_cast<size_t>(box.label) >= names2.size()) {
      continue;
    }
    const auto &cls = names2[static_cast<size_t>(box.label)];
    if (!tooth_detection_config.second_detection_labels.count(cls)) {
      continue;
    }

    box.left = x + box.left * sx;
    box.right = x + box.right * sx;
    box.top = y + box.top * sy;
    box.bottom = y + box.bottom * sy;

    box.left =
        std::max(0.0f, std::min(box.left, static_cast<float>(width - 1)));
    box.right =
        std::max(0.0f, std::min(box.right, static_cast<float>(width - 1)));
    box.top = std::max(0.0f, std::min(box.top, static_cast<float>(height - 1)));
    box.bottom =
        std::max(0.0f, std::min(box.bottom, static_cast<float>(height - 1)));

    out_boxes.push_back(box);

    if (tooth_detection_config.tooth_state.count(box.label)) {
      tooth_detection_config.tooth_state_name.emplace(cls);
      if (tooth_detection_config.excavator_type ==
          ExcavatorType::Front_shovel_excavator) {
        tooth_detection_config.teeth_root_coordinates.emplace_back(
            std::make_tuple(box.label, static_cast<int>(box.left - 50),
                            static_cast<int>(box.left - 50)));
        tooth_detection_config.teeth_root_coordinates.emplace_back(
            std::make_tuple(box.label, static_cast<int>(box.right + 50),
                            static_cast<int>(box.right + 50)));
      }
    } else if (tooth_detection_config.teeth_and_root.count(box.label)) {
      tooth_detection_config.teeth_root_coordinates.emplace_back(
          std::make_tuple(box.label, static_cast<int>(box.left),
                          static_cast<int>(box.right)));
    }
  }
}
} // namespace nv12_bucket_second_detection
