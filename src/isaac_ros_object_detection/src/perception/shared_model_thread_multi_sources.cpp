#include "shared_model_thread_multi_sources.h"
#include "distortion_correction/undistort_factory.h"
#include "gxf_nv12_convert.h"
#include "gxf_nv12_draw.h"
#include "nv12_bucket_second_detection.h"
#include "yolo_detect.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string_view>
#include <tuple>
#include <unordered_map>

namespace {
bool EnsureCudaBuffer(void **buffer, size_t &buffer_size, size_t required_size,
                      rclcpp::Logger logger, const char *tag) {
  if (*buffer && buffer_size >= required_size) {
    return true;
  }
  if (*buffer) {
    cudaFree(*buffer);
    *buffer = nullptr;
    buffer_size = 0;
  }
  cudaError_t err = cudaMalloc(buffer, required_size);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(logger, "Failed to allocate CUDA buffer for %s: %s", tag,
                 cudaGetErrorString(err));
    return false;
  }
  buffer_size = required_size;
  return true;
}

void SplitPrimaryBoxesByPeopleThreshold(const std::vector<Box> &primary_boxes,
                                        int people_label, float threshold,
                                        std::vector<Box> &people_boxes,
                                        std::vector<Box> &non_people_boxes,
                                        int &people_cnt) {
  people_boxes.clear();
  non_people_boxes.clear();
  people_cnt = 0;
  for (const auto &b : primary_boxes) {
    if (people_label >= 0 && b.label == people_label) {
      if (b.confidence >= threshold) {
        people_boxes.push_back(b);
        people_cnt += 1;
      }
      continue;
    }
    non_people_boxes.push_back(b);
  }
}

void UpdateToothConfigFromPrimaryBoxes(
    const std::vector<Box> &boxes, const std::vector<std::string> &class_names,
    ToothDetectionConfig &tooth_detection_config) {
  for (const auto &box : boxes) {
    if (box.label < 0 || static_cast<size_t>(box.label) >= class_names.size()) {
      continue;
    }
    const auto &cls = class_names[static_cast<size_t>(box.label)];
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

bool IsDetectionEnabledByMqtt(std::unique_ptr<MQTTPublish> &mqtt_publisher,
                              std::mutex &mqtt_mtx) {
  std::lock_guard<std::mutex> lock(mqtt_mtx);
  if (!mqtt_publisher) {
    return true;
  }
  return mqtt_publisher->is_subscribe_enabled();
}

bool PublishMqttJson(std::unique_ptr<MQTTPublish> &mqtt_publisher,
                     std::mutex &mqtt_mtx, const nlohmann::json &payload_json) {
  std::lock_guard<std::mutex> lock(mqtt_mtx);
  if (!mqtt_publisher || !mqtt_publisher->isMqttConnected()) {
    return false;
  }
  return mqtt_publisher->publishAsync(payload_json.dump());
}

void ResetMqttAlarmsIfNeeded(
    ToothDetectionConfig &tooth_detection_config,
    PedestrianDetectionConfig &pedestrian_detection_config,
    std::unique_ptr<MQTTPublish> &mqtt_publisher, std::mutex &mqtt_mtx) {
  if (pedestrian_detection_config.is_people_detected) {
    nlohmann::json payload_json;
    payload_json["peopleCount"] = 0;
    if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json)) {
      pedestrian_detection_config.last_people_detected_time_ =
          std::chrono::steady_clock::time_point();
      pedestrian_detection_config.is_people_detected = false;
      pedestrian_detection_config.show_people_cnt = 0;
    }
  }

  if (tooth_detection_config.is_tooth_detected) {
    nlohmann::json payload_json;
    payload_json["toothDetection"] =
        tooth_detection_config.detection_mapping["complete"];
    if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json)) {
      tooth_detection_config.last_tooth_detected_time_ =
          std::chrono::steady_clock::time_point();
      tooth_detection_config.is_tooth_detected = false;
      tooth_detection_config.show_tooth_state_name = "complete";
    }
  }
}

void HandlePeopleMqtt(ToothDetectionConfig &tooth_detection_config,
                      PedestrianDetectionConfig &pedestrian_detection_config,
                      std::unique_ptr<MQTTPublish> &mqtt_publisher,
                      std::mutex &mqtt_mtx) {
  (void)tooth_detection_config;
  pedestrian_detection_config.show_people_cnt =
      pedestrian_detection_config.people_cnt;
  if (pedestrian_detection_config.show_people_cnt > 0) {
    nlohmann::json payload_json;
    payload_json["peopleCount"] = pedestrian_detection_config.show_people_cnt;
    if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json)) {
      pedestrian_detection_config.last_people_detected_time_ =
          std::chrono::steady_clock::now();
      pedestrian_detection_config.is_people_detected = true;
    }
  }

  if (pedestrian_detection_config.is_people_detected) {
    const auto current_time = std::chrono::steady_clock::now();
    if (pedestrian_detection_config.last_people_detected_time_ !=
            std::chrono::steady_clock::time_point() &&
        (current_time -
         pedestrian_detection_config.last_people_detected_time_) >
            std::chrono::seconds(1)) {
      if (pedestrian_detection_config.show_people_cnt == 0) {
        nlohmann::json payload_json;
        payload_json["peopleCount"] = 0;
        if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json)) {
          pedestrian_detection_config.last_people_detected_time_ =
              std::chrono::steady_clock::time_point();
          pedestrian_detection_config.is_people_detected = false;
        }
      }
    }
  }
}

void HandleToothMqtt(ToothDetectionConfig &tooth_detection_config,
                     const std::vector<std::string> &class_names_for_root,
                     std::unique_ptr<MQTTPublish> &mqtt_publisher,
                     std::mutex &mqtt_mtx) {
  if (tooth_detection_config.tooth_state_name.size() == 1) {
    const std::string state = *tooth_detection_config.tooth_state_name.begin();
    if (tooth_detection_config.excavator_type ==
        ExcavatorType::Back_shovel_excavator) {
      if (static_cast<int>(
              tooth_detection_config.teeth_root_coordinates.size()) ==
          tooth_detection_config.teeth_complete_number) {
        std::sort(tooth_detection_config.teeth_root_coordinates.begin(),
                  tooth_detection_config.teeth_root_coordinates.end(),
                  [&](const std::tuple<int, int, int> &a,
                      const std::tuple<int, int, int> &b) {
                    return std::get<1>(a) < std::get<1>(b);
                  });
        tooth_detection_config.root_idx.clear();
        for (int i = 0;
             i < static_cast<int>(
                     tooth_detection_config.teeth_root_coordinates.size());
             ++i) {
          const int lbl =
              std::get<0>(tooth_detection_config.teeth_root_coordinates[i]);
          if (lbl >= 0 &&
              static_cast<size_t>(lbl) < class_names_for_root.size() &&
              class_names_for_root[static_cast<size_t>(lbl)] == "root") {
            tooth_detection_config.root_idx.emplace_back(i + 1);
          }
        }

        if (tooth_detection_config.root_idx.empty() && state == "complete") {
          ++tooth_detection_config.detect_res["complete"];
          ++tooth_detection_config.match_cnt;
          if (tooth_detection_config.detect_res["complete"] >
              tooth_detection_config.max_count) {
            tooth_detection_config.max_count =
                tooth_detection_config.detect_res["complete"];
            tooth_detection_config.max_class_label = "complete";
          }
        } else {
          for (const auto &idx : tooth_detection_config.root_idx) {
            const std::string label = "miss" + std::to_string(idx);
            if (state == label) {
              ++tooth_detection_config.detect_res[label];
              ++tooth_detection_config.match_cnt;
              if (tooth_detection_config.detect_res[label] >
                  tooth_detection_config.max_count) {
                tooth_detection_config.max_count =
                    tooth_detection_config.detect_res[label];
                tooth_detection_config.max_class_label = label;
              }
            }
          }
        }
      }
    } else {
      if (static_cast<int>(
              tooth_detection_config.teeth_root_coordinates.size()) ==
              tooth_detection_config.teeth_complete_number + 2 &&
          state == "complete") {
        ++tooth_detection_config.detect_res["complete"];
        ++tooth_detection_config.match_cnt;
        if (tooth_detection_config.detect_res["complete"] >
            tooth_detection_config.max_count) {
          tooth_detection_config.max_count =
              tooth_detection_config.detect_res["complete"];
          tooth_detection_config.max_class_label = "complete";
        }
      } else {
        std::sort(tooth_detection_config.teeth_root_coordinates.begin(),
                  tooth_detection_config.teeth_root_coordinates.end(),
                  [&](const std::tuple<int, int, int> &a,
                      const std::tuple<int, int, int> &b) {
                    return std::get<1>(a) < std::get<1>(b);
                  });
        tooth_detection_config.root_idx.clear();
        for (int i = 0;
             i + 1 < static_cast<int>(
                         tooth_detection_config.teeth_root_coordinates.size());
             ++i) {
          const int dist = std::abs(
              std::get<1>(
                  tooth_detection_config.teeth_root_coordinates[i + 1]) -
              std::get<1>(tooth_detection_config.teeth_root_coordinates[i]));
          tooth_detection_config.root_idx.emplace_back(dist);
        }
        if (!tooth_detection_config.root_idx.empty()) {
          const auto max_it =
              std::max_element(tooth_detection_config.root_idx.begin(),
                               tooth_detection_config.root_idx.end());
          const auto max_gap_idx = static_cast<int>(
              max_it - tooth_detection_config.root_idx.begin());
          const std::string label = "miss" + std::to_string(max_gap_idx + 1);
          if (state == label) {
            ++tooth_detection_config.detect_res[label];
            ++tooth_detection_config.match_cnt;
            if (tooth_detection_config.detect_res[label] >
                tooth_detection_config.max_count) {
              tooth_detection_config.max_count =
                  tooth_detection_config.detect_res[label];
              tooth_detection_config.max_class_label = label;
            }
          }
        }
      }
    }

    if (tooth_detection_config.match_cnt > 0 &&
        tooth_detection_config.match_cnt % 5 == 0 &&
        tooth_detection_config.max_count >
            static_cast<int>(tooth_detection_config.match_cnt * 0.7)) {
      tooth_detection_config.show_tooth_state_name =
          tooth_detection_config.max_class_label;
      if (tooth_detection_config.max_class_label != "complete") {
        nlohmann::json payload_json;
        payload_json["toothDetection"] =
            tooth_detection_config
                .detection_mapping[tooth_detection_config.max_class_label];
        if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json)) {
          tooth_detection_config.last_tooth_detected_time_ =
              std::chrono::steady_clock::now();
          tooth_detection_config.is_tooth_detected = true;
        }
      }
      tooth_detection_config.match_cnt = 0;
      tooth_detection_config.max_count = 0;
      tooth_detection_config.max_class_label.clear();
      tooth_detection_config.detect_res.clear();
    }
  } else {
    tooth_detection_config.show_tooth_state_name.clear();
  }

  if (tooth_detection_config.is_tooth_detected) {
    const auto current_time = std::chrono::steady_clock::now();
    if (tooth_detection_config.last_tooth_detected_time_ !=
            std::chrono::steady_clock::time_point() &&
        (current_time - tooth_detection_config.last_tooth_detected_time_) >
            std::chrono::seconds(1)) {
      if (tooth_detection_config.teeth_root_coordinates.empty()) {
        nlohmann::json payload_json;
        payload_json["toothDetection"] =
            tooth_detection_config.detection_mapping["complete"];
        if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json)) {
          tooth_detection_config.last_tooth_detected_time_ =
              std::chrono::steady_clock::time_point();
          tooth_detection_config.is_tooth_detected = false;
        }
      }
    }
  }
}
} // namespace

// NITROS/GXF: 为 NV12 发布构建自定义 entity（支持 buffer 池回收）
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wpedantic"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"
#pragma GCC diagnostic pop
#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"

namespace {
constexpr uint64_t kNsPerSecLocal = 1000000000ULL;

static nvidia::isaac_ros::nitros::NitrosImage
BuildNv12NitrosImageWithPool(const std_msgs::msg::Header &header,
                             uint32_t height, uint32_t width, void *gpu_data,
                             int y_stride, int uv_stride,
                             const std::shared_ptr<Nv12PublishPool> &pool) {
  auto message = nvidia::gxf::Entity::New(
      nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext());
  if (!message) {
    std::stringstream error_msg;
    error_msg << "[BuildNv12NitrosImageWithPool] Error initializing entity: "
              << GxfResultStr(message.error());
    throw std::runtime_error(error_msg.str().c_str());
  }

  nvidia::isaac_ros::nitros::NitrosImage nitros_image{};
  nitros_image.handle = message->eid();
  nitros_image.frame_id = header.frame_id;
  GxfEntityRefCountInc(
      nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext(),
      message->eid());

  auto output_timestamp = message->add<nvidia::gxf::Timestamp>("timestamp");
  if (!output_timestamp) {
    std::stringstream error_msg;
    error_msg << "[BuildNv12NitrosImageWithPool] Failed to add timestamp: "
              << GxfResultStr(output_timestamp.error());
    throw std::runtime_error(error_msg.str().c_str());
  }
  output_timestamp.value()->acqtime =
      static_cast<uint64_t>(header.stamp.sec) * kNsPerSecLocal +
      static_cast<uint64_t>(header.stamp.nanosec);

  auto gxf_image =
      message->add<nvidia::gxf::VideoBuffer>(header.frame_id.c_str());
  if (!gxf_image) {
    std::stringstream error_msg;
    error_msg << "[BuildNv12NitrosImageWithPool] Failed to add VideoBuffer: "
              << GxfResultStr(gxf_image.error());
    throw std::runtime_error(error_msg.str().c_str());
  }

  if (width % 2 != 0 || height % 2 != 0) {
    throw std::runtime_error(
        "[BuildNv12NitrosImageWithPool] width/height must be even for NV12");
  }

  // IMPORTANT:
  // 不要假设 NV12 的 stride。NITROS/GXF 的 VideoBuffer 可能使用对齐后的
  // stride， 且本工程内部 NV12 约定为
  // 2-plane：Y(bytes_per_pixel=1)、UV(bytes_per_pixel=2) 通过显式传入
  // y_stride/uv_stride 来保持 buffer 的 layout 与 metadata 一致，
  // 避免出现“绿色蒙层”等色度错位问题。
  using GxfVideoFormat = nvidia::gxf::VideoFormat;
  nvidia::gxf::VideoFormatSize<GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER>
      format_size;
  std::array<nvidia::gxf::ColorPlane, 2> planes{
      nvidia::gxf::ColorPlane("Y", 1, static_cast<uint32_t>(y_stride)),
      nvidia::gxf::ColorPlane("UV", 2, static_cast<uint32_t>(uv_stride)),
  };
  const uint64_t size =
      format_size.size(width, height, planes, /*stride_align=*/false);
  std::vector<nvidia::gxf::ColorPlane> color_planes{planes.begin(),
                                                    planes.end()};

  constexpr auto surface_layout =
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  constexpr auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;

  nvidia::gxf::VideoBufferInfo buffer_info{
      width, height, GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER,
      std::move(color_planes), surface_layout};

  gxf_image.value()->wrapMemory(buffer_info, size, storage_type, gpu_data,
                                [pool](void *ptr) {
                                  if (pool) {
                                    pool->release(ptr);
                                    return nvidia::gxf::Success;
                                  }
                                  cudaFree(ptr);
                                  return nvidia::gxf::Success;
                                });

  return nitros_image;
}
} // namespace

// ==================== NITROS 图像处理函数 ====================

// 将 BGR cv::Mat 转换为 NV12 并发布为 NITROS 消息（使用 CUDA kernel，绕过 VPI）
bool SharedModelMultiSourcesPipelineManager::publishNitrosImage(
    InputContext &ctx, const cv::Mat &bgr_frame, int source_id) {
  if (bgr_frame.empty()) {
    RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                 "Source %d: publishNitrosImage - bgr_frame is empty!",
                 source_id);
    return false;
  }
  if (!ctx.nitros_image_publisher_) {
    RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                 "Source %d: publishNitrosImage - publisher is null!",
                 source_id);
    return false;
  }

  const int width = bgr_frame.cols;
  const int height = bgr_frame.rows;

  // GXF NV12 格式参数
  const int y_stride = width;       // Y 平面 stride
  const int uv_stride = width * 2;  // UV 平面 stride（GXF 格式）
  const int bgr_stride = width * 3; // BGR stride

  // 复用 BGR GPU 缓冲，避免每帧 cudaMalloc/cudaFree
  const size_t bgr_size = static_cast<size_t>(bgr_stride) * height;
  if (!EnsureCudaBuffer(&ctx.cuda_bgr_publish_buffer_,
                        ctx.cuda_bgr_publish_buffer_size_, bgr_size,
                        ros2_client_->getNode()->get_logger(), "publish BGR")) {
    return false;
  }

  // 为 NV12 发布准备/复用发布池（与 NV12 直通路径保持一致）
  const size_t nv12_size = static_cast<size_t>(width) * height * 2;
  if (!ctx.nitros_publish_pool_ ||
      ctx.nitros_publish_pool_->size_bytes != nv12_size) {
    ctx.nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
    ctx.nitros_publish_pool_->warmup(6);
  }
  auto pool = ctx.nitros_publish_pool_;

  void *publish_buffer = pool ? pool->acquire() : nullptr;
  if (!publish_buffer) {
    RCLCPP_ERROR(
        ros2_client_->getNode()->get_logger(),
        "Source %d: publishNitrosImage - failed to acquire NV12 buffer",
        source_id);
    return false;
  }

  try {
    // 将 BGR 数据上传到 GPU（异步，使用每源独立 stream）
    cudaError_t cuda_err =
        cudaMemcpyAsync(ctx.cuda_bgr_publish_buffer_, bgr_frame.data, bgr_size,
                        cudaMemcpyHostToDevice, ctx.cuda_stream_);
    if (cuda_err != cudaSuccess) {
      pool->release(publish_buffer);
      throw std::runtime_error("Failed to copy BGR to GPU");
    }

    // BGR -> GXF NV12（GPU->GPU，异步提交到每源 stream）
    cuda_err = ConvertBGRToGxfNV12(publish_buffer, ctx.cuda_bgr_publish_buffer_,
                                   width, height, y_stride, uv_stride,
                                   bgr_stride, ctx.cuda_stream_);
    if (cuda_err != cudaSuccess) {
      pool->release(publish_buffer);
      throw std::runtime_error("ConvertBGRToGxfNV12 failed");
    }

    // 等待当前源 stream 完成（避免全局 cudaDeviceSynchronize）
    cuda_err = cudaStreamSynchronize(ctx.cuda_stream_);
    if (cuda_err != cudaSuccess) {
      pool->release(publish_buffer);
      throw std::runtime_error("cudaStreamSynchronize failed");
    }

    std_msgs::msg::Header header;
    header.stamp = ros2_client_->getNode()->now();
    header.frame_id =
        "camera_" + std::to_string(ctx.input_config_.camera_ids[source_id]);

    try {
      auto nitros_image = BuildNv12NitrosImageWithPool(
          header, static_cast<uint32_t>(height), static_cast<uint32_t>(width),
          publish_buffer, y_stride, uv_stride, pool);
      ctx.nitros_image_publisher_->publish(nitros_image);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                   "Source %d: build/publish nitros image failed: %s",
                   source_id, e.what());
      pool->release(publish_buffer);
      return false;
    }

    RCLCPP_DEBUG(ros2_client_->getNode()->get_logger(),
                 "Source %d: Published NITROS NV12 image %dx%d", source_id,
                 width, height);

  } catch (const std::exception &e) {
    RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                 "Failed to publish NITROS image: %s", e.what());
    return false;
  }

  return true;
}

// Image Raw 解码处理线程（sensor_msgs::msg::Image）
void SharedModelMultiSourcesPipelineManager::processImageRawDecoding(
    int source_id) {
  auto &ctx = input_contexts_[source_id];
  const cv::Size &input_size = ctx.input_config_.input_sizes[source_id];

  while (is_running_.load(std::memory_order_acquire)) {
    std::vector<uint8_t> image_data;
    {
      std::unique_lock<std::mutex> lock(ctx.decode_queue_mutex_);
      ctx.decode_cond_.wait(lock, [&]() {
        return !ctx.decode_queue_.empty() ||
               !is_running_.load(std::memory_order_acquire);
      });
      if (!is_running_.load(std::memory_order_acquire))
        break;

      if (ctx.decode_queue_.empty())
        continue;

      image_data = std::move(ctx.decode_queue_.front());
      ctx.decode_queue_.pop();
    }

    // 假设输入是 BGR24 或 RGB24 格式
    cv::Mat img;
    const size_t expected_size =
        static_cast<size_t>(input_size.width) * input_size.height * 3;

    if (image_data.size() == expected_size) {
      // BGR24/RGB24 格式
      img = cv::Mat(input_size.height, input_size.width, CV_8UC3,
                    image_data.data())
                .clone();
    } else {
      RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                   "Source %d: Unexpected image data size %zu, expected %zu",
                   source_id, image_data.size(), expected_size);
      continue;
    }

    if (img.empty()) {
      RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                   "Source %d: Image decoding failed", source_id);
      continue;
    }

    enqueueRawFrame(source_id, std::move(img));
  }
}

// ==================== 核心管理函数 ====================

SharedModelMultiSourcesPipelineManager::SharedModelMultiSourcesPipelineManager()
    : SharedModelMultiSourcesPipelineManager(InputConfig{}) {}

SharedModelMultiSourcesPipelineManager::SharedModelMultiSourcesPipelineManager(
    const InputConfig &input_config) {
  this->input_config_ = input_config;
  if (!input_config_.use_ros2_component_container) {
    initResources();
  }
}

SharedModelMultiSourcesPipelineManager::
    ~SharedModelMultiSourcesPipelineManager() {
  if (is_running_.load(std::memory_order_acquire)) {
    stop();
  }
}

bool SharedModelMultiSourcesPipelineManager::isRunning() const {
  return is_running_.load(std::memory_order_acquire);
}

void SharedModelMultiSourcesPipelineManager::initResources() {
  // 获取输入源数量
  // 优先从 camera_namespaces 获取（用于 NITROS 模式）
  // 否则从传统配置获取
  if (input_config_.use_ros2_component_container) {
    this->num_sources = input_config_.camera_namespaces.size();
    sample::gLogInfo << "Using camera namespaces for input sources"
                     << std::endl;
  } else {
    this->num_sources = std::max({input_config_.image_paths.size(),
                                  input_config_.video_paths.size(),
                                  input_config_.camera_ids.size()});
    sample::gLogInfo << "Using image paths and video paths for input sources"
                     << std::endl;
  }

  if (num_sources == 0) {
    sample::gLogError << "No input sources configured." << std::endl;
    return;
  }

  // 初始化轮询检测调度用的 ready 标志
  source_ready_flags_.assign(num_sources, false);

  sample::gLogInfo << "Initializing perception for " << num_sources
                   << " camera sources" << std::endl;

  // 确保 camera_ids 和 input_sizes 数组大小正确
  if (input_config_.camera_ids.size() < num_sources) {
    input_config_.camera_ids.resize(num_sources);
    for (size_t i = 0; i < num_sources; ++i) {
      input_config_.camera_ids[i] = static_cast<int>(i);
    }
  }

  if (input_config_.input_sizes.size() < num_sources) {
    cv::Size default_size(1920, 1536);
    if (!input_config_.input_sizes.empty()) {
      default_size = input_config_.input_sizes[0];
    }
    input_config_.input_sizes.resize(num_sources, default_size);
  }

  if (roi_config_.roi_regions.size() < num_sources) {
    std::vector<int> default_roi = {0, 0, 1920, 1536};
    if (!roi_config_.roi_regions.empty()) {
      default_roi = roi_config_.roi_regions[0];
    }
    roi_config_.roi_regions.resize(num_sources, default_roi);
  }

  const cv::Size &first_size = input_config_.input_sizes[0];
  const std::vector<int> &first_region = roi_config_.roi_regions[0];

  bool all_sizes_equal = true;
  bool all_regions_equal = true;

  for (size_t i = 1; i < num_sources; ++i) {
    if (input_config_.input_sizes[i] != first_size) {
      all_sizes_equal = false;
      break;
    }
    if (roi_config_.roi_regions[i] != first_region) {
      all_regions_equal = false;
      break;
    }
  }

  if (!all_sizes_equal || !all_regions_equal) {
    sample::gLogError
        << "Input sizes or ROI regions are not consistent across all sources."
        << std::endl;
    return;
  }

  if (!general_config_.is_init) {
    sample::gLogInfo << "Input source: "
                     << input_config_.input_source[static_cast<uint8_t>(
                            input_config_.source)]
                     << std::endl;

    // 初始化共享模型资源
    {
      std::lock_guard<std::mutex> lock(yolo_mutex_);
      yolo_ = std::make_unique<YOLOModel>(
          image_processing_config_, input_config_, model_config_, roi_config_);
    }

    if (model_config_.need_second_detection_) {
      std::lock_guard<std::mutex> lock(second_yolo_mutex_);
      second_yolo_ = std::make_unique<YOLOModel>(
          image_processing_config_, input_config_, model_config_, roi_config_);
      model_config_.bucket_thread_pool_ = std::make_shared<ThreadPool>(
          model_config_.bucket_detection_thread_limit);
      sample::gLogInfo << "Second YOLO model initialized." << std::endl;
    }

    if (model_config_.need_classifier_) {
      std::lock_guard<std::mutex> lock(trt_classifier_mutex_);
      trt_classifier_ = std::make_unique<TensorRTInference>(model_config_);
      if (!trt_classifier_ || !trt_classifier_->isReady()) {
        trt_classifier_.reset();
      }
      model_config_.classifier_thread_pool_ =
          std::make_shared<ThreadPool>(model_config_.classifier_thread_limit);
    }

    {
      std::lock_guard<std::mutex> lock(mqtt_publisher_mutex_);
      mqtt_publisher_ = std::make_unique<MQTTPublish>(mqtt_config_);
    }

    // 初始化ROS2
    if (input_config_.ros2_enabled) {
      if (input_config_.use_ros2_component_container) {
        // 组件模式
        {
          std::lock_guard<std::mutex> lock(ros2_client_mutex_);
          if (ros2_client_ == nullptr) {
            ros2_client_ = std::make_unique<ROS2Client>(
                parent_node_, input_config_, num_sources);
          }
        }
        sample::gLogInfo
            << "ROS2 initialized in component mode using parent node."
            << std::endl;
      } else {
        // 独立模式
        if (!rclcpp::ok()) {
          rclcpp::init(0, nullptr);
        }

        size_t num_ros_threads =
            std::min(std::thread::hardware_concurrency(),
                     static_cast<unsigned int>(this->num_sources));
        executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(
            rclcpp::ExecutorOptions(), num_ros_threads);

        {
          std::lock_guard<std::mutex> lock(ros2_client_mutex_);
          if (ros2_client_ == nullptr) {
            ros2_client_ = std::make_unique<ROS2Client>(
                input_config_, "yolo_ros2_node", num_sources);
            executor_->add_node(ros2_client_->get_node_base_interface());
          }
        }

        sample::gLogInfo << "ROS2 node initialized with multi-threaded "
                            "executor in standalone mode."
                         << std::endl;
      }

      // 为每个源创建独立回调组
      callback_groups_.reserve(num_sources);
      for (size_t i = 0; i < num_sources; ++i) {
        auto group = ros2_client_->getNode()->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callback_groups_.push_back(group);
      }

      // 注意：订阅者的创建移动到 input_contexts_ 填充之后
      // 以避免回调函数在上下文初始化之前被触发的竞态条件
    }

    general_config_.is_init = true;
  }

  // 预分配上下文容器空间
  input_contexts_.reserve(num_sources);

  use_nitros_sub_ = (input_config_.ros2_enabled &&
                     image_encoder_decoder_config_.encoding_format_sub ==
                         EncodingFormat::NITROS_NV12);

  for (size_t i = 0; i < num_sources; ++i) {
    InputConfig local_input_config = input_config_;
    local_input_config.idx = static_cast<int>(i);
    ModelConfig local_model_config = model_config_;
    ToothDetectionConfig local_tooth_config = tooth_detection_config_;
    PedestrianDetectionConfig local_pedestrian_config =
        pedestrian_detection_config_;
    ImageProcessingConfig local_image_config = image_processing_config_;
    ROIConfig local_roi_config = roi_config_;
    MqttConfig local_mqtt_config = mqtt_config_;
    UndistortConfig local_undistort_config = undistort_config_;
    ImageEncoderDecoderConfig local_image_encoder_decoder_config =
        image_encoder_decoder_config_;

    input_contexts_.emplace(
        std::piecewise_construct, std::forward_as_tuple(static_cast<int>(i)),
        std::forward_as_tuple(local_model_config, local_input_config,
                              local_tooth_config, local_pedestrian_config,
                              local_image_config, local_roi_config,
                              local_mqtt_config, local_undistort_config,
                              local_image_encoder_decoder_config));

    auto &ctx = input_contexts_[i];

    // 预分配 NITROS NV12 池槽位（避免首帧 resize 竞争；显存 buffer
    // 仍在回调里按需 malloc）
    if (use_nitros_sub_ && ctx.nitros_nv12_pool_.empty()) {
      ctx.nitros_nv12_pool_.resize(NITROS_NV12_POOL_SLOTS);
    }

    // 创建发布器
    if (ctx.input_config_.ros2_enabled && ros2_client_) {
      const bool use_nitros_pub =
          (ctx.image_encoder_decoder_config_.encoding_format_pub ==
           EncodingFormat::NITROS_NV12);

      if (use_nitros_pub) {
        // NITROS NV12 协商话题发布
        std::string pub_topic_name =
            ctx.input_config_.nitros_pub_topic_prefix.empty()
                ? ("/" + ctx.input_config_.camera_namespaces[i] +
                   "/image_perceptual")
                : ctx.input_config_.nitros_pub_topic_prefix + "/" +
                      ctx.input_config_.camera_namespaces[i] +
                      "/image_perceptual";

        ctx.nitros_image_publisher_ = std::make_shared<
            nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                nvidia::isaac_ros::nitros::NitrosImage>>(
            ros2_client_->getNode().get(), pub_topic_name,
            nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name,
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
            rclcpp::QoS(1));

        sample::gLogInfo << "Created NITROS NV12 publisher for source " << i
                         << " on topic: " << pub_topic_name << std::endl;
      } else {
        // sensor_msgs::msg::Image 发布
        ctx.image_raw_publisher_ =
            ros2_client_->getNode()->create_publisher<sensor_msgs::msg::Image>(
                "/" + ctx.input_config_.camera_namespaces[i] +
                    "/image_perceptual/raw",
                rclcpp::QoS(1).reliable());

        sample::gLogInfo << "Created Image publisher for source " << i
                         << std::endl;
      }
    }

    // 初始化输入源（非ROS2模式）
    if (!ctx.input_config_.ros2_enabled &&
        !openInputStream(ctx.image_processing_config_, ctx.input_config_)) {
      sample::gLogError << "Failed to open input stream for source " << i
                        << std::endl;
      return;
    }
  }

  // ========== 创建订阅者（在 input_contexts_ 初始化之后）==========
  // 这样可以避免回调函数在上下文初始化之前被触发的竞态条件
  if (input_config_.ros2_enabled && ros2_client_) {
    // 设置QoS策略：低延迟优先（只保留最新一帧 + best_effort）
    rclcpp::QoS qos_profile(1);
    qos_profile.best_effort();
    qos_profile.keep_last(1);

    // 检查订阅格式
    if (use_nitros_sub_) {
      // ========== NITROS NV12 协商话题订阅 ==========
      sample::gLogInfo << "Using NITROS NV12 subscription format" << std::endl;
      nitros_subscribers_.reserve(num_sources);

      for (size_t i = 0; i < num_sources; ++i) {
        auto source_id = static_cast<int>(i);
        std::string topic_name =
            input_config_.nitros_topic_prefix.empty()
                ? ("/" + input_config_.camera_namespaces[i] + "/image_raw")
                : input_config_.nitros_topic_prefix + "/" +
                      input_config_.camera_namespaces[i] + "/image_raw";

        auto nitros_sub = std::make_shared<
            nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                nvidia::isaac_ros::nitros::NitrosImageView>>(
            ros2_client_->getNode().get(), topic_name,
            nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name,
            std::function<void(
                const nvidia::isaac_ros::nitros::NitrosImageView &)>(
                [this, source_id](
                    const nvidia::isaac_ros::nitros::NitrosImageView &view) {
                  // 检查上下文是否存在
                  auto it = this->input_contexts_.find(source_id);
                  if (it == this->input_contexts_.end()) {
                    return;
                  }
                  auto &ctx = it->second;

                  const int width = view.GetWidth();
                  const int height = view.GetHeight();
                  const unsigned char *gpu_data = view.GetGpuData();
                  const int32_t stamp_sec = view.GetTimestampSeconds();
                  const uint32_t stamp_nanosec = view.GetTimestampNanoseconds();

                  if (!gpu_data || width <= 0 || height <= 0) {
                    return;
                  }

                  // 读取真实 stride/size（不要假设）
                  const int y_stride = static_cast<int>(view.GetStride(0));
                  const int uv_stride = static_cast<int>(view.GetStride(1));
                  const size_t total_size =
                      static_cast<size_t>(view.GetSizeInBytes());

                  // 从“预分配池”获取一个可用槽位，避免每帧 cudaMalloc/cudaFree
                  // 抖动
                  void *local_gpu_buffer = nullptr;
                  cudaEvent_t ready_event = nullptr;
                  int pool_index = -1;
                  {
                    std::lock_guard<std::mutex> pool_lock(
                        ctx.nitros_nv12_pool_mutex_);
                    if (ctx.nitros_nv12_pool_.empty()) {
                      ctx.nitros_nv12_pool_.resize(NITROS_NV12_POOL_SLOTS);
                    }
                    // 选择一个可用槽位：优先找 in_use=false；否则找 event 已
                    // ready 的
                    for (size_t attempt = 0;
                         attempt < ctx.nitros_nv12_pool_.size(); ++attempt) {
                      const int idx = static_cast<int>(attempt);
                      auto &slot = ctx.nitros_nv12_pool_[idx];
                      // 注意：就算 slot 的 ready_event 已
                      // ready，只要该槽位对应的帧
                      // 仍可能在队列/解码线程中被使用，就绝不能复用，否则会覆盖数据。
                      if (slot.in_use) {
                        continue;
                      }

                      if (slot.gpu_buffer && slot.size_bytes != total_size) {
                        cudaFree(slot.gpu_buffer);
                        slot.gpu_buffer = nullptr;
                        slot.size_bytes = 0;
                      }

                      if (!slot.gpu_buffer) {
                        cudaError_t aerr =
                            cudaMalloc(&slot.gpu_buffer, total_size);
                        if (aerr != cudaSuccess) {
                          RCLCPP_ERROR(
                              ros2_client_->getNode()->get_logger(),
                              "Source %d: cudaMalloc(pool) failed in callback",
                              source_id);
                          return;
                        }
                        slot.size_bytes = total_size;
                      }
                      if (!slot.ready_event) {
                        cudaError_t eerr = cudaEventCreateWithFlags(
                            &slot.ready_event, cudaEventDisableTiming);
                        if (eerr != cudaSuccess) {
                          RCLCPP_ERROR(
                              ros2_client_->getNode()->get_logger(),
                              "Source %d: cudaEventCreate(pool) failed",
                              source_id);
                          return;
                        }
                      }

                      slot.in_use = true;
                      local_gpu_buffer = slot.gpu_buffer;
                      ready_event = slot.ready_event;
                      pool_index = idx;
                      break;
                    }
                  }

                  if (!local_gpu_buffer || !ready_event || pool_index < 0) {
                    // 池满：低延迟策略下直接丢帧
                    return;
                  }

                  // 异步提交到该源的 stream，不在回调里做同步阻塞
                  cudaError_t err = cudaMemcpyAsync(
                      local_gpu_buffer, gpu_data, total_size,
                      cudaMemcpyDeviceToDevice, ctx.cuda_stream_);
                  if (err != cudaSuccess) {
                    RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                                 "Source %d: cudaMemcpyAsync failed",
                                 source_id);
                    // 归还槽位（copy 未提交成功）
                    std::lock_guard<std::mutex> pool_lock(
                        ctx.nitros_nv12_pool_mutex_);
                    if (static_cast<size_t>(pool_index) <
                        ctx.nitros_nv12_pool_.size()) {
                      ctx.nitros_nv12_pool_[pool_index].in_use = false;
                    }
                    return;
                  }
                  err = cudaEventRecord(ready_event, ctx.cuda_stream_);
                  if (err != cudaSuccess) {
                    RCLCPP_ERROR(ros2_client_->getNode()->get_logger(),
                                 "Source %d: cudaEventRecord failed",
                                 source_id);
                    // 归还槽位（event 未记录成功）
                    std::lock_guard<std::mutex> pool_lock(
                        ctx.nitros_nv12_pool_mutex_);
                    if (static_cast<size_t>(pool_index) <
                        ctx.nitros_nv12_pool_.size()) {
                      ctx.nitros_nv12_pool_[pool_index].in_use = false;
                    }
                    return;
                  }

                  // 该槽位会同时被“发布线程”和“检测线程”消费
                  {
                    std::lock_guard<std::mutex> pool_lock(
                        ctx.nitros_nv12_pool_mutex_);
                    if (static_cast<size_t>(pool_index) <
                        ctx.nitros_nv12_pool_.size()) {
                      ctx.nitros_nv12_pool_[pool_index].ref_count = 2;
                    }
                  }

                  // // [DEBUG] 订阅回调
                  // RCLCPP_INFO_THROTTLE(
                  //     ros2_client_->getNode()->get_logger(),
                  //     *ros2_client_->getNode()->get_clock(), 2000,
                  //     "Source %d: NITROS callback - GPU copy done, "
                  //     "width=%d height=%d",
                  //     source_id, width, height);

                  auto release_ref = [&](InputContext &ctx,
                                         const InputContext::GpuNV12Data &d) {
                    if (d.pool_index >= 0) {
                      bool push_gc = false;
                      {
                        std::lock_guard<std::mutex> pool_lock(
                            ctx.nitros_nv12_pool_mutex_);
                        if (static_cast<size_t>(d.pool_index) <
                            ctx.nitros_nv12_pool_.size()) {
                          auto &slot = ctx.nitros_nv12_pool_[d.pool_index];
                          if (slot.ref_count > 0) {
                            slot.ref_count -= 1;
                          }
                          if (slot.ref_count == 0) {
                            push_gc = true;
                          }
                        }
                      }
                      if (push_gc) {
                        std::lock_guard<std::mutex> gc_lock(
                            ctx.nitros_nv12_pool_gc_mutex_);
                        ctx.nitros_nv12_pool_gc_.push(d.pool_index);
                      }
                    } else {
                      std::lock_guard<std::mutex> gc_lock(
                          ctx.nitros_gpu_gc_mutex_);
                      ctx.nitros_gpu_gc_queue_.push(d);
                    }
                  };

                  // 1) 发布队列：每帧发布，没检测到就叠历史框
                  {
                    std::lock_guard<std::mutex> lock(
                        ctx.nitros_gpu_queue_mutex_);
                    while (ctx.nitros_gpu_queue_.size() >= 1) {
                      auto old = ctx.nitros_gpu_queue_.front();
                      ctx.nitros_gpu_queue_.pop();
                      release_ref(ctx, old);
                    }
                    ctx.nitros_gpu_queue_.push({local_gpu_buffer, width, height,
                                                y_stride, uv_stride, total_size,
                                                ready_event, pool_index,
                                                stamp_sec, stamp_nanosec});
                  }
                  ctx.nitros_gpu_cond_.notify_one();

                  // 2) 检测队列：轮询检测（只保留最新）
                  {
                    std::lock_guard<std::mutex> lock(
                        ctx.nitros_detect_queue_mutex_);
                    while (ctx.nitros_detect_queue_.size() >= 1) {
                      auto old = ctx.nitros_detect_queue_.front();
                      ctx.nitros_detect_queue_.pop();
                      release_ref(ctx, old);
                    }
                    ctx.nitros_detect_queue_.push(
                        {local_gpu_buffer, width, height, y_stride, uv_stride,
                         total_size, ready_event, pool_index, stamp_sec,
                         stamp_nanosec});
                    ctx.last_detection_enqueue_tp_ =
                        std::chrono::steady_clock::now();
                    ctx.has_last_detection_enqueue_tp_ = true;
                  }

                  // NV12 直推理路径：通知全局检测线程该源有新帧
                  markSourceReadyForDetection(source_id);
                }),
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, qos_profile);

        nitros_subscribers_.push_back(nitros_sub);
        sample::gLogInfo << "Created NITROS subscriber for source " << i
                         << " on topic: " << topic_name << std::endl;
      }
    } else {
      // ========== sensor_msgs::msg::Image 订阅 ==========
      sample::gLogInfo << "Using sensor_msgs::msg::Image subscription format"
                       << std::endl;
      image_raw_subscribers_.reserve(num_sources);
      for (size_t i = 0; i < num_sources; ++i) {
        auto source_id = static_cast<int>(i);
        std::string topic_name =
            input_config_.nitros_topic_prefix.empty()
                ? ("/" + input_config_.camera_namespaces[i] + "/image_raw")
                : input_config_.nitros_topic_prefix + "/" +
                      input_config_.camera_namespaces[i] + "/image_raw";

        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = callback_groups_[i];
        auto sub =
            ros2_client_->getNode()
                ->create_subscription<sensor_msgs::msg::Image>(
                    topic_name, qos_profile,
                    [this,
                     source_id](const sensor_msgs::msg::Image::SharedPtr msg) {
                      if (!msg) {
                        return;
                      }
                      auto it = this->input_contexts_.find(source_id);
                      if (it == this->input_contexts_.end()) {
                        return;
                      }

                      cv::Mat bgr;
                      const int width = static_cast<int>(msg->width);
                      const int height = static_cast<int>(msg->height);
                      if (width <= 0 || height <= 0) {
                        return;
                      }

                      if (msg->encoding == "bgr8") {
                        bgr = cv::Mat(height, width, CV_8UC3,
                                      const_cast<uint8_t *>(msg->data.data()),
                                      msg->step)
                                  .clone();
                      } else if (msg->encoding == "rgb8") {
                        cv::Mat rgb(height, width, CV_8UC3,
                                    const_cast<uint8_t *>(msg->data.data()),
                                    msg->step);
                        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                      } else if (msg->encoding == "bgra8") {
                        cv::Mat bgra(height, width, CV_8UC4,
                                     const_cast<uint8_t *>(msg->data.data()),
                                     msg->step);
                        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
                      } else if (msg->encoding == "rgba8") {
                        cv::Mat rgba(height, width, CV_8UC4,
                                     const_cast<uint8_t *>(msg->data.data()),
                                     msg->step);
                        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                      } else if (msg->encoding == "yuv422_yuy2" ||
                                 msg->encoding == "yuyv" ||
                                 msg->encoding == "yuy2") {
                        cv::Mat yuy2(height, width, CV_8UC2,
                                     const_cast<uint8_t *>(msg->data.data()),
                                     msg->step);
                        cv::cvtColor(yuy2, bgr, cv::COLOR_YUV2BGR_YUY2);
                      } else if (msg->encoding == "nv12") {
                        const int yuv_rows = height + (height / 2);
                        cv::Mat yuv(yuv_rows, width, CV_8UC1,
                                    const_cast<uint8_t *>(msg->data.data()),
                                    msg->step);
                        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
                      } else if (msg->encoding == "nv21") {
                        const int yuv_rows = height + (height / 2);
                        cv::Mat yuv(yuv_rows, width, CV_8UC1,
                                    const_cast<uint8_t *>(msg->data.data()),
                                    msg->step);
                        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);
                      } else {
                        return;
                      }

                      if (bgr.empty()) {
                        return;
                      }

                      enqueueRawFrame(source_id, std::move(bgr));
                    },
                    sub_options);
        image_raw_subscribers_.push_back(sub);
        sample::gLogInfo << "Created Image subscriber for source " << i
                         << " on topic: " << topic_name << std::endl;
      }
    }
  }

  // 为每个源创建独立的去畸变器
  for (auto &[source_id, ctx] : input_contexts_) {
    if (ctx.undistort_config_.need_undistort_) {
      ctx.undistorter_ =
          createUndistorter(ctx.undistort_config_, ctx.input_config_);
    }
  }
}

void SharedModelMultiSourcesPipelineManager::cleanupResources() {
  // 清空队列
  for (auto &[source_id, ctx] : input_contexts_) {
    {
      std::lock_guard lock(ctx.raw_frame_mutex_);
      std::queue<std::unique_ptr<cv::Mat>> empty_queue;
      ctx.raw_frame_queue.swap(empty_queue);
    }
    {
      std::lock_guard lock(ctx.detection_mutex_);
      std::queue<std::unique_ptr<cv::Mat>> empty_queue;
      ctx.detection_queue.swap(empty_queue);
    }
    {
      std::lock_guard lock(ctx.display_mutex_);
      std::queue<cv::Mat> empty_queue;
      ctx.display_queue.swap(empty_queue);
    }
    {
      std::lock_guard lock(ctx.decode_queue_mutex_);
      std::queue<std::vector<uint8_t>> empty_queue;
      ctx.decode_queue_.swap(empty_queue);
    }
    {
      // 清理 GPU NV12 队列（需要释放 GPU 缓冲区）
      std::lock_guard lock(ctx.nitros_gpu_queue_mutex_);
      while (!ctx.nitros_gpu_queue_.empty()) {
        auto &data = ctx.nitros_gpu_queue_.front();
        if (data.pool_index >= 0) {
          std::lock_guard<std::mutex> pool_lock(ctx.nitros_nv12_pool_mutex_);
          if (static_cast<size_t>(data.pool_index) <
              ctx.nitros_nv12_pool_.size()) {
            ctx.nitros_nv12_pool_[data.pool_index].in_use = false;
          }
        } else {
          if (data.ready_event) {
            cudaEventDestroy(data.ready_event);
            data.ready_event = nullptr;
          }
          if (data.gpu_buffer) {
            cudaFree(data.gpu_buffer);
          }
        }
        ctx.nitros_gpu_queue_.pop();
      }
    }
    {
      // 清理延迟回收队列（回调丢弃的 GPU buffer/event）
      std::lock_guard lock(ctx.nitros_gpu_gc_mutex_);
      while (!ctx.nitros_gpu_gc_queue_.empty()) {
        auto &data = ctx.nitros_gpu_gc_queue_.front();
        if (data.ready_event) {
          cudaEventDestroy(data.ready_event);
          data.ready_event = nullptr;
        }
        if (data.gpu_buffer) {
          cudaFree(data.gpu_buffer);
        }
        ctx.nitros_gpu_gc_queue_.pop();
      }
    }
    {
      // 清理 NV12 池 GC 队列
      std::lock_guard lock(ctx.nitros_nv12_pool_gc_mutex_);
      std::queue<int> empty;
      ctx.nitros_nv12_pool_gc_.swap(empty);
    }
    {
      // 释放 NV12 池资源（buffer + event）
      std::lock_guard<std::mutex> pool_lock(ctx.nitros_nv12_pool_mutex_);
      for (auto &slot : ctx.nitros_nv12_pool_) {
        if (slot.ready_event) {
          cudaEventDestroy(slot.ready_event);
          slot.ready_event = nullptr;
        }
        if (slot.gpu_buffer) {
          cudaFree(slot.gpu_buffer);
          slot.gpu_buffer = nullptr;
        }
        slot.size_bytes = 0;
        slot.in_use = false;
      }
      ctx.nitros_nv12_pool_.clear();
    }
  }

  // 释放发布器
  for (auto &[source_id, ctx] : input_contexts_) {
    {
      std::lock_guard<std::mutex> lock(ctx.publish_mutex_);
      ctx.nitros_image_publisher_.reset();
      ctx.image_raw_publisher_.reset();
    }
    {
      std::lock_guard<std::mutex> lock(ctx.undistorter_mutex_);
      ctx.undistorter_.reset();
    }
  }

  {
    std::lock_guard<std::mutex> lock(ros2_client_mutex_);
    ros2_client_.reset();
  }
  {
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    yolo_.reset();
  }
  {
    std::lock_guard<std::mutex> lock(second_yolo_mutex_);
    second_yolo_.reset();
    model_config_.bucket_thread_pool_.reset();
  }
  {
    std::lock_guard<std::mutex> lock(trt_classifier_mutex_);
    trt_classifier_.reset();
    model_config_.classifier_thread_pool_.reset();
  }
  {
    std::lock_guard<std::mutex> lock(mqtt_publisher_mutex_);
    mqtt_publisher_.reset();
  }

  if (input_config_.ros2_enabled && rclcpp::ok() && !parent_node_) {
    rclcpp::shutdown();
  }

  input_contexts_.clear();

  sample::gLogInfo << "All threads and resources have been cleaned up."
                   << std::endl;
}

void SharedModelMultiSourcesPipelineManager::enqueueRawFrame(
    int source_id, const cv::Mat &frame) {
  auto &ctx = input_contexts_[source_id];
  std::lock_guard<std::mutex> lock(ctx.raw_frame_mutex_);
  if (ctx.raw_frame_queue.size() >= MAX_QUEUE_SIZE) {
    ctx.raw_frame_queue.pop();
  }
  ctx.raw_frame_queue.emplace(std::make_unique<cv::Mat>(frame.clone()));
  ctx.raw_frame_cond_.notify_one();
}

void SharedModelMultiSourcesPipelineManager::enqueueRawFrame(int source_id,
                                                             cv::Mat &&frame) {
  auto &ctx = input_contexts_[source_id];
  std::lock_guard<std::mutex> lock(ctx.raw_frame_mutex_);
  if (ctx.raw_frame_queue.size() >= MAX_QUEUE_SIZE) {
    ctx.raw_frame_queue.pop();
  }
  ctx.raw_frame_queue.emplace(std::make_unique<cv::Mat>(std::move(frame)));
  ctx.raw_frame_cond_.notify_one();
}

void SharedModelMultiSourcesPipelineManager::markSourceReadyForDetection(
    int source_id) {
  if (source_id < 0)
    return;
  const auto sid = static_cast<size_t>(source_id);
  {
    std::lock_guard<std::mutex> lock(ready_mutex_);
    if (sid >= source_ready_flags_.size()) {
      return;
    }
    source_ready_flags_[sid] = true;
  }
  ready_cond_.notify_one();
}

bool SharedModelMultiSourcesPipelineManager::anySourceReadyForDetectionLocked()
    const {
  for (bool v : source_ready_flags_) {
    if (v)
      return true;
  }
  return false;
}

cv::Mat
SharedModelMultiSourcesPipelineManager::dequeueDisplayFrame(int source_id) {
  auto &ctx = input_contexts_[source_id];
  std::lock_guard<std::mutex> lock(ctx.display_mutex_);
  if (ctx.display_queue.empty()) {
    return cv::Mat();
  }
  cv::Mat frame = std::move(ctx.display_queue.front());
  ctx.display_queue.pop();
  return frame;
}

void SharedModelMultiSourcesPipelineManager::GUI() {
  cv::namedWindow("Multi-Source Display",
                  cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
  cv::resizeWindow("Multi-Source Display", general_config_.combined.cols,
                   general_config_.combined.rows);

  int grid_rows = 1, grid_cols = 1;
  if (num_sources == 2) {
    grid_rows = 1;
    grid_cols = 2;
  } else if (num_sources >= 3) {
    grid_rows = 2;
    grid_cols = 2;
  }
  const int sub_width = general_config_.combined.cols / grid_cols;
  const int sub_height = general_config_.combined.rows / grid_rows;

  while (is_running_.load(std::memory_order_acquire)) {
    for (const auto &[source_id, _] : input_contexts_) {
      cv::Mat frame = dequeueDisplayFrame(source_id);
      if (frame.empty()) {
        frame = cv::Mat::zeros(sub_height, sub_width, CV_8UC3);
      } else {
        cv::resize(frame, frame, cv::Size(sub_width, sub_height));
      }

      const int row = source_id / grid_cols;
      const int col = source_id % grid_cols;
      cv::Rect roi(col * sub_width, row * sub_height, sub_width, sub_height);

      frame.copyTo(general_config_.combined(roi));

      cv::putText(general_config_.combined,
                  "Source " + std::to_string(source_id),
                  cv::Point(col * sub_width + 10, row * sub_height + 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Multi-Source Display", general_config_.combined);
    int key = cv::waitKey(1);
    if (key == 27 || !is_running_.load(std::memory_order_acquire)) {
      is_running_.store(false, std::memory_order_release);
      for (auto &[id, ctx] : input_contexts_) {
        ctx.raw_frame_cond_.notify_all();
        ctx.detection_cond_.notify_all();
        ctx.display_cond_.notify_all();
        ctx.decode_cond_.notify_all();
      }
      break;
    }
  }

  cv::destroyWindow("Multi-Source Display");
}

void SharedModelMultiSourcesPipelineManager::processUndistortAndDraw(
    int source_id) {
  auto &ctx = input_contexts_[source_id];

  const rclcpp::Logger logger = ros2_client_
                                    ? ros2_client_->getNode()->get_logger()
                                    : rclcpp::get_logger("object_detection");
  const auto clock = ros2_client_
                         ? ros2_client_->getNode()->get_clock()
                         : std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);

  RCLCPP_INFO(logger, "Source %d: processUndistortAndDraw thread started",
              source_id);

  // ===== NV12 订阅输入：每帧叠历史框并发布（保证帧率）=====
  if (use_nitros_sub_) {
    while (is_running_.load(std::memory_order_acquire)) {
      const auto loop_start = std::chrono::high_resolution_clock::now();
      // 机会性回收：ref_count==0 且 event ready 的槽位归还
      {
        std::lock_guard<std::mutex> pool_lock(ctx.nitros_nv12_pool_mutex_);
        std::lock_guard<std::mutex> gc_lock(ctx.nitros_nv12_pool_gc_mutex_);
        while (!ctx.nitros_nv12_pool_gc_.empty()) {
          const int idx = ctx.nitros_nv12_pool_gc_.front();
          if (idx < 0 ||
              static_cast<size_t>(idx) >= ctx.nitros_nv12_pool_.size()) {
            ctx.nitros_nv12_pool_gc_.pop();
            continue;
          }
          auto &slot = ctx.nitros_nv12_pool_[idx];
          if (slot.ref_count != 0) {
            // 仍在被引用，不应回收
            ctx.nitros_nv12_pool_gc_.pop();
            continue;
          }
          if (!slot.ready_event) {
            slot.in_use = false;
            ctx.nitros_nv12_pool_gc_.pop();
            continue;
          }
          const auto q = cudaEventQuery(slot.ready_event);
          if (q == cudaErrorNotReady) {
            break;
          }
          slot.in_use = false;
          ctx.nitros_nv12_pool_gc_.pop();
        }
      }

      InputContext::GpuNV12Data gpu_data = {nullptr, 0, 0,       0,
                                            0,       0, nullptr, -1};
      {
        std::unique_lock<std::mutex> lock(ctx.nitros_gpu_queue_mutex_);
        ctx.nitros_gpu_cond_.wait(lock, [&]() {
          return !ctx.nitros_gpu_queue_.empty() ||
                 !is_running_.load(std::memory_order_acquire);
        });
        if (!is_running_.load(std::memory_order_acquire)) {
          break;
        }
        while (ctx.nitros_gpu_queue_.size() > 1) {
          // 丢弃旧帧：释放发布引用
          auto old = ctx.nitros_gpu_queue_.front();
          ctx.nitros_gpu_queue_.pop();
          if (old.pool_index >= 0) {
            bool push_gc = false;
            {
              std::lock_guard<std::mutex> pool_lock(
                  ctx.nitros_nv12_pool_mutex_);
              if (static_cast<size_t>(old.pool_index) <
                  ctx.nitros_nv12_pool_.size()) {
                auto &slot = ctx.nitros_nv12_pool_[old.pool_index];
                if (slot.ref_count > 0) {
                  slot.ref_count -= 1;
                }
                if (slot.ref_count == 0) {
                  push_gc = true;
                }
              }
            }
            if (push_gc) {
              std::lock_guard<std::mutex> gc_lock(
                  ctx.nitros_nv12_pool_gc_mutex_);
              ctx.nitros_nv12_pool_gc_.push(old.pool_index);
            }
          } else {
            std::lock_guard<std::mutex> gc_lock(ctx.nitros_gpu_gc_mutex_);
            ctx.nitros_gpu_gc_queue_.push(old);
          }
        }
        if (ctx.nitros_gpu_queue_.empty()) {
          continue;
        }
        gpu_data = ctx.nitros_gpu_queue_.front();
        ctx.nitros_gpu_queue_.pop();
      }

      if (!gpu_data.gpu_buffer || gpu_data.width <= 0 || gpu_data.height <= 0 ||
          !ctx.nitros_image_publisher_) {
        // 释放发布引用
        if (gpu_data.pool_index >= 0) {
          bool push_gc = false;
          {
            std::lock_guard<std::mutex> pool_lock(ctx.nitros_nv12_pool_mutex_);
            if (static_cast<size_t>(gpu_data.pool_index) <
                ctx.nitros_nv12_pool_.size()) {
              auto &slot = ctx.nitros_nv12_pool_[gpu_data.pool_index];
              if (slot.ref_count > 0) {
                slot.ref_count -= 1;
              }
              if (slot.ref_count == 0) {
                push_gc = true;
              }
            }
          }
          if (push_gc) {
            std::lock_guard<std::mutex> gc_lock(ctx.nitros_nv12_pool_gc_mutex_);
            ctx.nitros_nv12_pool_gc_.push(gpu_data.pool_index);
          }
        } else if (gpu_data.gpu_buffer) {
          std::lock_guard<std::mutex> gc_lock(ctx.nitros_gpu_gc_mutex_);
          ctx.nitros_gpu_gc_queue_.push(gpu_data);
        }
        continue;
      }

      // 计算 FPS（以“发布频率”为准）
      {
        const auto now_tp = std::chrono::steady_clock::now();
        ctx.fps_window_frames_ += 1;
        const double sec =
            std::chrono::duration<double>(now_tp - ctx.fps_window_start_tp_)
                .count();
        if (sec >= 1.0) {
          ctx.fps_value_ = static_cast<double>(ctx.fps_window_frames_) / sec;
          ctx.fps_window_frames_ = 0;
          ctx.fps_window_start_tp_ = now_tp;
        }
      }
      const int fps_x10 = static_cast<int>(std::lround(ctx.fps_value_ * 10.0));

      const size_t nv12_size =
          gpu_data.total_size > 0
              ? gpu_data.total_size
              : (static_cast<size_t>(gpu_data.width) * gpu_data.height * 2);

      // 为当前分辨率选择/创建发布池（旧池可能仍被在途消息持有，shared_ptr
      // 会延迟释放）
      if (!ctx.nitros_publish_pool_ ||
          ctx.nitros_publish_pool_->size_bytes != nv12_size) {
        ctx.nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
        // 预热：避免运行时 cudaMalloc 抖动（经验值：每路 6 个 buffer
        // 足够覆盖在途消息）
        ctx.nitros_publish_pool_->warmup(6);
      }
      auto pool = ctx.nitros_publish_pool_;

      void *publish_buffer = pool ? pool->acquire() : nullptr;
      cudaError_t perr =
          (publish_buffer ? cudaSuccess : cudaErrorMemoryAllocation);
      if (perr == cudaSuccess && publish_buffer) {
        if (gpu_data.ready_event) {
          cudaStreamWaitEvent(ctx.cuda_publish_stream_, gpu_data.ready_event,
                              0);
        }
        perr =
            cudaMemcpyAsync(publish_buffer, gpu_data.gpu_buffer, nv12_size,
                            cudaMemcpyDeviceToDevice, ctx.cuda_publish_stream_);

        auto &people_bbs = ctx.nv12_people_bbs_host_;
        auto &marks = ctx.nv12_marks_host_;
        people_bbs.clear();
        marks.clear();
        {
          std::lock_guard<std::mutex> last_lock(ctx.last_detection_mutex);
          if (!ctx.last_detection_boxes.empty()) {
            const auto &boxes0 = ctx.last_detection_boxes[0];
            people_bbs.reserve(boxes0.size());
            marks.reserve(boxes0.size());
            for (const auto &b : boxes0) {
              if (b.label < 0) {
                continue;
              }

              if (b.label == 0) {
                people_bbs.push_back({static_cast<int>(std::lround(b.left)),
                                      static_cast<int>(std::lround(b.top)),
                                      static_cast<int>(std::lround(b.right)),
                                      static_cast<int>(std::lround(b.bottom))});
                continue;
              }

              uint8_t yv = 0;
              uint8_t uv = 0;
              uint8_t vv = 0;
              if (b.label == 2) {
                yv = 145;
                uv = 54;
                vv = 34;
              } else if (b.label == 3) {
                yv = 100;
                uv = 90;
                vv = 240;
              } else {
                continue;
              }

              const int cx = static_cast<int>(std::lround(
                  (static_cast<double>(b.left) + static_cast<double>(b.right)) *
                  0.5));
              const int cy = static_cast<int>(std::lround(
                  (static_cast<double>(b.top) + static_cast<double>(b.bottom)) *
                  0.5));

              const int x = std::max(0, std::min(cx, gpu_data.width - 1));
              const int y = std::max(0, std::min(cy, gpu_data.height - 1));

              marks.push_back({x, y, 5, yv, uv, vv});
            }
          }
        }
        if (perr == cudaSuccess && !people_bbs.empty()) {
          if (EnsureCudaBuffer(
                  &ctx.cuda_bbox_buffer_, ctx.cuda_bbox_buffer_size_,
                  people_bbs.size() * sizeof(GxfBBox), logger, "bbox buffer")) {
            perr = cudaMemcpyAsync(ctx.cuda_bbox_buffer_, people_bbs.data(),
                                   people_bbs.size() * sizeof(GxfBBox),
                                   cudaMemcpyHostToDevice,
                                   ctx.cuda_publish_stream_);
            if (perr == cudaSuccess) {
              constexpr uint8_t y_val = 100;
              constexpr uint8_t u_val = 90;
              constexpr uint8_t v_val = 240;
              perr = DrawGxfNV12Bboxes(
                  publish_buffer, gpu_data.width, gpu_data.height,
                  gpu_data.y_stride, gpu_data.uv_stride,
                  static_cast<const GxfBBox *>(ctx.cuda_bbox_buffer_),
                  static_cast<int>(people_bbs.size()), 2, y_val, u_val, v_val,
                  ctx.cuda_publish_stream_);
            }
          }
        }

        if (perr == cudaSuccess && !marks.empty()) {
          if (EnsureCudaBuffer(&ctx.cuda_bbox_buffer_,
                               ctx.cuda_bbox_buffer_size_,
                               marks.size() * sizeof(GxfCircleMark), logger,
                               "circle marks buffer")) {
            perr = cudaMemcpyAsync(ctx.cuda_bbox_buffer_, marks.data(),
                                   marks.size() * sizeof(GxfCircleMark),
                                   cudaMemcpyHostToDevice,
                                   ctx.cuda_publish_stream_);
            if (perr == cudaSuccess) {
              perr = DrawGxfNV12CircleMarks(
                  publish_buffer, gpu_data.width, gpu_data.height,
                  gpu_data.y_stride, gpu_data.uv_stride,
                  static_cast<const GxfCircleMark *>(ctx.cuda_bbox_buffer_),
                  static_cast<int>(marks.size()), ctx.cuda_publish_stream_);
            }
          }
        }

        if (perr == cudaSuccess) {
          perr = DrawGxfNV12Fps(publish_buffer, gpu_data.width, gpu_data.height,
                                gpu_data.y_stride, gpu_data.uv_stride, fps_x10,
                                ctx.cuda_publish_stream_);
        }

        if (perr == cudaSuccess) {
          perr = cudaStreamSynchronize(ctx.cuda_publish_stream_);
        }
        if (perr == cudaSuccess) {
          std_msgs::msg::Header header;
          header.stamp.sec = gpu_data.stamp_sec;
          header.stamp.nanosec = gpu_data.stamp_nanosec;
          header.frame_id =
              "camera_" +
              std::to_string(
                  ctx.input_config_.camera_ids[ctx.input_config_.idx]);
          // {
          //   const int64_t now_ns =
          //   ros2_client_->getNode()->now().nanoseconds(); const int64_t
          //   msg_ns =
          //       static_cast<int64_t>(gpu_data.stamp_sec) * 1000000000LL +
          //       static_cast<int64_t>(gpu_data.stamp_nanosec);
          //   if (msg_ns > 0 && now_ns >= msg_ns) {
          //     const double age_ms =
          //         static_cast<double>(now_ns - msg_ns) / 1000000.0;
          //     RCLCPP_INFO_THROTTLE(logger, *clock, 1000,
          //                          "[PIPE] Age source=%d age=%.3fms",
          //                          source_id, age_ms);
          //   }
          // }
          try {
            auto nitros_image = BuildNv12NitrosImageWithPool(
                header, static_cast<uint32_t>(gpu_data.height),
                static_cast<uint32_t>(gpu_data.width), publish_buffer,
                gpu_data.y_stride, gpu_data.uv_stride, pool);
            ctx.nitros_image_publisher_->publish(nitros_image);
          } catch (const std::exception &e) {
            RCLCPP_ERROR(logger,
                         "Source %d: build/publish nitros image failed: %s",
                         source_id, e.what());
            if (pool) {
              pool->release(publish_buffer);
            } else {
              cudaFree(publish_buffer);
            }
          }
        } else {
          if (pool) {
            pool->release(publish_buffer);
          } else {
            cudaFree(publish_buffer);
          }
        }
      }

      // // 输出“总体耗时”（对齐直接打开相机日志），避免刷屏用 throttle
      // const auto loop_end = std::chrono::high_resolution_clock::now();
      // const auto total_ms =
      //     std::chrono::duration_cast<std::chrono::milliseconds>(loop_end -
      //                                                           loop_start)
      //         .count();
      // RCLCPP_INFO_THROTTLE(logger, *clock, 2000,
      //                      "[PIPE] Total Time source=%d total=%ldms",
      //                      source_id, static_cast<long>(total_ms));

      // 释放发布引用
      if (gpu_data.pool_index >= 0) {
        bool push_gc = false;
        {
          std::lock_guard<std::mutex> pool_lock(ctx.nitros_nv12_pool_mutex_);
          if (static_cast<size_t>(gpu_data.pool_index) <
              ctx.nitros_nv12_pool_.size()) {
            auto &slot = ctx.nitros_nv12_pool_[gpu_data.pool_index];
            if (slot.ref_count > 0) {
              slot.ref_count -= 1;
            }
            if (slot.ref_count == 0) {
              push_gc = true;
            }
          }
        }
        if (push_gc) {
          std::lock_guard<std::mutex> gc_lock(ctx.nitros_nv12_pool_gc_mutex_);
          ctx.nitros_nv12_pool_gc_.push(gpu_data.pool_index);
        }
      } else {
        std::lock_guard<std::mutex> gc_lock(ctx.nitros_gpu_gc_mutex_);
        ctx.nitros_gpu_gc_queue_.push(gpu_data);
      }
    }
    return;
  }

  // ======= 普通输入源：相机、视频等 =======
  while (is_running_.load(std::memory_order_acquire)) {
    if (!is_running_.load(std::memory_order_acquire))
      break;

    auto wait_start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<cv::Mat> raw_img;
    {
      std::unique_lock<std::mutex> lock(ctx.raw_frame_mutex_);
      ctx.raw_frame_cond_.wait(lock, [&]() {
        return !ctx.raw_frame_queue.empty() ||
               !is_running_.load(std::memory_order_acquire);
      });
      if (!is_running_.load(std::memory_order_acquire))
        break;

      // 只保留最新帧，降低队列累积带来的延迟
      while (ctx.raw_frame_queue.size() > 1) {
        ctx.raw_frame_queue.pop();
      }
      raw_img = std::move(ctx.raw_frame_queue.front());
      ctx.raw_frame_queue.pop();
    }
    auto wait_end = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::high_resolution_clock::now();

    // [DEBUG] 成功从队列取出图像
    RCLCPP_INFO_THROTTLE(logger, *clock, 2000,
                         "Source %d: processUndistortAndDraw got frame %dx%d",
                         source_id, raw_img ? raw_img->cols : 0,
                         raw_img ? raw_img->rows : 0);

    // 去畸变
    cv::Mat undistorted_img;
    {
      std::lock_guard<std::mutex> undistorter_lock(ctx.undistorter_mutex_);
      undistorted_img = ctx.undistort_config_.need_undistort_
                            ? ctx.undistorter_->undistort_image(*raw_img)
                            : *raw_img;
    }

    // 放入检测队列
    {
      std::lock_guard<std::mutex> lock(ctx.detection_mutex_);
      while (ctx.detection_queue.size() >= 1) {
        ctx.detection_queue.pop();
      }
      ctx.detection_queue.push(std::make_unique<cv::Mat>(undistorted_img));
      ctx.last_detection_enqueue_tp_ = std::chrono::steady_clock::now();
      ctx.has_last_detection_enqueue_tp_ = true;
      ctx.detection_cond_.notify_one();
    }
    // 通知全局检测线程：该源有新帧可检测（替代忙轮询）
    markSourceReadyForDetection(source_id);

    // 绘制检测框
    cv::Mat fusion_img;
    // 关键优化：绘制使用本地
    // ImageProcessingConfig，避免与检测线程抢占同一把锁
    // （检测线程只在轮到时检测，未轮到时这里直接使用历史框，不等待检测完成）
    ImageProcessingConfig local_cfg;
    {
      std::lock_guard<std::mutex> lock(ctx.image_processing_mutex);
      local_cfg = ctx.image_processing_config_;
    }
    local_cfg.init_frame = undistorted_img;
    {
      std::lock_guard<std::mutex> last_lock(ctx.last_detection_mutex);
      local_cfg.bboxes = ctx.last_detection_boxes;
    }

    if (local_cfg.bboxes.empty()) {
      fusion_img = local_cfg.init_frame;
    } else {
      fusion_img = draw_detection_boxes(
          yolo_->getParam().class_names, local_cfg, ctx.tooth_detection_config_,
          ctx.pedestrian_detection_config_, ctx.input_config_, ctx.roi_config_);
    }

    // FPS 统计是在 draw_detection_boxes() 内更新的；由于我们用的是
    // local_cfg， 需要把 FPS 相关字段写回 ctx，避免一直显示 0。
    {
      std::lock_guard<std::mutex> lock(ctx.image_processing_mutex);
      if (source_id >= 0 &&
          static_cast<size_t>(source_id) <
              ctx.image_processing_config_.fps_list.size() &&
          static_cast<size_t>(source_id) < local_cfg.fps_list.size()) {
        ctx.image_processing_config_.fps_list[source_id] =
            local_cfg.fps_list[source_id];
        ctx.image_processing_config_.frameCount_list[source_id] =
            local_cfg.frameCount_list[source_id];
        ctx.image_processing_config_.lastTime_list[source_id] =
            local_cfg.lastTime_list[source_id];
      }
    }

    // 发布图像（只支持 NITROS NV12 和 sensor_msgs::msg::Image）
    if (ctx.input_config_.ros2_enabled) {
      if (ctx.image_encoder_decoder_config_.encoding_format_pub ==
          EncodingFormat::NITROS_NV12) {
        // NITROS NV12 协商话题发布
        bool success = publishNitrosImage(ctx, fusion_img, source_id);
        if (success) {
          RCLCPP_DEBUG(logger, "Source %d: NITROS NV12 image published",
                       source_id);
        } else {
          RCLCPP_ERROR(logger, "Source %d: Failed to publish NITROS NV12 image",
                       source_id);
        }
      } else {
        // sensor_msgs::msg::Image 发布（BGR格式）
        const std::string frame_id =
            "camera_" + std::to_string(ctx.input_config_.camera_ids[source_id]);
        auto msg = std::make_unique<sensor_msgs::msg::Image>();
        msg->header.stamp = ros2_client_->getNode()->now();
        msg->header.frame_id = frame_id;
        msg->encoding = "bgr8";
        msg->height = fusion_img.rows;
        msg->width = fusion_img.cols;
        msg->step = fusion_img.cols * 3;
        msg->is_bigendian = false;
        msg->data.assign(fusion_img.data,
                         fusion_img.data +
                             fusion_img.total() * fusion_img.elemSize());

        {
          std::lock_guard<std::mutex> lock(ctx.publish_mutex_);
          if (ctx.image_raw_publisher_) {
            ctx.image_raw_publisher_->publish(std::move(msg));
          }
        }
      }
    }

    // 放入显示队列
    if (general_config_.show_gui) {
      std::lock_guard<std::mutex> lock(ctx.display_mutex_);
      std::queue<cv::Mat> empty_queue;
      ctx.display_queue.swap(empty_queue);
      ctx.display_queue.push(std::move(fusion_img));
      ctx.display_cond_.notify_one();
    }

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     end_time - start_time);
    // auto wait_duration =
    // std::chrono::duration_cast<std::chrono::milliseconds>(
    //     wait_end - wait_start);
    // (void)wait_duration;
    // RCLCPP_INFO_THROTTLE(logger, *clock, 2000,
    //                      "[PIPE] Total Time source=%d total=%ldms",
    //                      source_id, duration.count());
  }
}

void SharedModelMultiSourcesPipelineManager::processDetectionGlobal() {
  size_t current_source = 0;
  const rclcpp::Logger logger = ros2_client_
                                    ? ros2_client_->getNode()->get_logger()
                                    : rclcpp::get_logger("object_detection");
  const auto clock = ros2_client_
                         ? ros2_client_->getNode()->get_clock()
                         : std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  while (is_running_.load(std::memory_order_acquire)) {
    if (!is_running_.load(std::memory_order_acquire)) {
      break;
    }

    // 阻塞等待：直到任一 source 有新帧可检测（替代每路 wait_for + 忙轮询）
    size_t selected_source = current_source;
    {
      std::unique_lock<std::mutex> lk(ready_mutex_);
      ready_cond_.wait(lk, [&]() {
        return !is_running_.load(std::memory_order_acquire) ||
               anySourceReadyForDetectionLocked();
      });
      if (!is_running_.load(std::memory_order_acquire)) {
        break;
      }

      // 公平调度（round-robin）：从 current_source 起找下一个 ready 的源
      bool found = false;
      for (size_t i = 0; i < num_sources; ++i) {
        const size_t s = (current_source + i) % num_sources;
        if (s < source_ready_flags_.size() && source_ready_flags_[s]) {
          selected_source = s;
          source_ready_flags_[s] = false; // 消费该“ready”标志
          found = true;
          break;
        }
      }
      if (!found) {
        // 极端竞态：被唤醒但没有 ready（例如 stop 通知），回到循环
        continue;
      }
    }

    if (use_nitros_sub_) {
      // ========== NITROS NV12：micro-batch 推理（提升总吞吐、降低
      // wait）==========
      const int max_batch =
          yolo_ ? static_cast<int>(yolo_->getParam().batch_size) : 1;

      std::vector<size_t> batch_sources;
      batch_sources.reserve(std::max(1, max_batch));
      std::vector<InputContext::GpuNV12Data> batch_frames;
      batch_frames.reserve(std::max(1, max_batch));
      std::vector<std::chrono::steady_clock::time_point> batch_enq_tps;
      batch_enq_tps.reserve(std::max(1, max_batch));
      std::vector<bool> batch_has_enq_tps;
      batch_has_enq_tps.reserve(std::max(1, max_batch));

      auto clear_ready_flag = [&](size_t sid) {
        std::lock_guard<std::mutex> lk(ready_mutex_);
        if (sid < source_ready_flags_.size()) {
          source_ready_flags_[sid] = false;
        }
      };

      auto try_take_one = [&](size_t sid) -> bool {
        auto &c = input_contexts_[sid];

        // 机会性回收：回调线程丢弃的“池槽位”，等事件 ready 后归还
        {
          std::lock_guard<std::mutex> pool_lock(c.nitros_nv12_pool_mutex_);
          std::lock_guard<std::mutex> gc_lock(c.nitros_nv12_pool_gc_mutex_);
          while (!c.nitros_nv12_pool_gc_.empty()) {
            const int idx = c.nitros_nv12_pool_gc_.front();
            if (idx < 0 ||
                static_cast<size_t>(idx) >= c.nitros_nv12_pool_.size()) {
              c.nitros_nv12_pool_gc_.pop();
              continue;
            }
            auto &slot = c.nitros_nv12_pool_[idx];
            if (!slot.ready_event) {
              slot.in_use = false;
              c.nitros_nv12_pool_gc_.pop();
              continue;
            }
            const auto q = cudaEventQuery(slot.ready_event);
            if (q == cudaErrorNotReady) {
              break;
            }
            slot.in_use = false;
            c.nitros_nv12_pool_gc_.pop();
          }
        }

        InputContext::GpuNV12Data d = {nullptr, 0, 0, 0, 0, 0, nullptr, -1};
        std::chrono::steady_clock::time_point enq_tp;
        bool has_enq_tp = false;
        {
          std::lock_guard<std::mutex> lock(c.nitros_detect_queue_mutex_);
          while (c.nitros_detect_queue_.size() > 1) {
            c.nitros_detect_queue_.pop();
          }
          if (!c.nitros_detect_queue_.empty()) {
            d = c.nitros_detect_queue_.front();
            c.nitros_detect_queue_.pop();
            c.detection_dequeued_ += 1;
          } else {
            c.ready_but_empty_ += 1;
          }
          has_enq_tp = c.has_last_detection_enqueue_tp_;
          enq_tp = c.last_detection_enqueue_tp_;
          c.has_last_detection_enqueue_tp_ = false;
        }

        // 无论是否取到帧，都清除 ready flag，避免无意义唤醒导致忙循环
        clear_ready_flag(sid);

        if (!d.gpu_buffer || d.width <= 0 || d.height <= 0) {
          return false;
        }
        batch_sources.push_back(sid);
        batch_frames.push_back(d);
        batch_enq_tps.push_back(enq_tp);
        batch_has_enq_tps.push_back(has_enq_tp);
        return true;
      };

      // 先尝试取当前被调度到的 source
      if (!try_take_one(selected_source)) {
        current_source = (selected_source + 1) % num_sources;
        continue;
      }

      // 尽量填满 batch（从 round-robin 顺序向后收集）
      for (size_t i = 1;
           i < num_sources && (int)batch_sources.size() < max_batch; ++i) {
        const size_t sid = (selected_source + i) % num_sources;
        (void)try_take_one(sid);
      }

      const auto detect_start_steady = std::chrono::steady_clock::now();

      const bool detection_enabled =
          IsDetectionEnabledByMqtt(mqtt_publisher_, mqtt_publisher_mutex_);
      if (!detection_enabled) {
        for (size_t bi = 0; bi < batch_sources.size(); ++bi) {
          const size_t sid = batch_sources[bi];
          auto &c = input_contexts_[sid];

          ResetMqttAlarmsIfNeeded(c.tooth_detection_config_,
                                  c.pedestrian_detection_config_,
                                  mqtt_publisher_, mqtt_publisher_mutex_);

          c.tooth_detection_config_.tooth_state_name.clear();
          c.tooth_detection_config_.teeth_root_coordinates.clear();
          c.tooth_detection_config_.root_idx.clear();
          c.pedestrian_detection_config_.people_cnt = 0;
          c.pedestrian_detection_config_.show_people_cnt = 0;

          std::vector<std::vector<Box>> bboxes;
          bboxes.resize(1);

          // const auto detect_end_steady = std::chrono::steady_clock::now();
          // const long detect_ms =
          //     std::chrono::duration_cast<std::chrono::milliseconds>(
          //         detect_end_steady - detect_start_steady)
          //         .count();
          // const long wait_ms =
          //     batch_has_enq_tps[bi]
          //         ? std::chrono::duration_cast<std::chrono::milliseconds>(
          //               detect_start_steady - batch_enq_tps[bi])
          //               .count()
          //         : -1;

          // const long total_ms = (wait_ms >= 0 ? wait_ms : 0) + detect_ms;
          // RCLCPP_INFO_THROTTLE(
          //     logger, *clock, 2000,
          //     "[DETECT] source=%zu detect=%ldms wait=%ldms total=%ldms", sid,
          //     detect_ms, wait_ms, total_ms);

          {
            std::lock_guard<std::mutex> last_lock(c.last_detection_mutex);
            c.last_detection_boxes = std::move(bboxes);
          }
          {
            std::lock_guard<std::mutex> lock(c.image_processing_mutex);
            c.image_processing_config_.bboxes = c.last_detection_boxes;
          }
        }
      } else {
        // ===========================
        // 第一次检测（主模型 YOLO）：
        // - NITROS/NV12 路径下，输入就是 GPU 上的 NV12（GXF layout）
        // - 这里做 micro-batch：把多路 source 的 NV12 指针塞进同一个 YOLO
        // 实例，
        //   用 batch 推理提升总体吞吐（需要 TRT 引擎支持 dynamic batch）
        // - y.copyFromGxfNv12Gpu 会在 GPU 上做 NV12->BGR 写入 YOLO
        // 的输入缓冲，
        //   然后 preprocess/infer/postprocess 都在同一条 stream 上跑
        //
        // 注意：共享模型只有一个 yolo_ 实例，必须用 global_detection_mutex_
        // 串行化，避免多线程同时访问同一个 TRT context / CUDA buffer。
        // ===========================
        std::vector<std::vector<Box>> batch_boxes;
        batch_boxes.resize(batch_sources.size());
        {
          std::lock_guard<std::mutex> global_lock(global_detection_mutex_);
          auto &y = yolo_->getYolo();
          const int run_batch = static_cast<int>(batch_sources.size());
          for (int bi = 0; bi < run_batch; ++bi) {
            const auto &d = batch_frames[bi];
            y.copyFromGxfNv12Gpu(d.gpu_buffer, d.width, d.height, d.y_stride,
                                 d.uv_stride, d.ready_event, bi);
          }
          y.preprocess();
          y.infer();
          y.postprocess(run_batch);
          const auto objectss = y.getObjectss();
          for (int bi = 0; bi < run_batch; ++bi) {
            if (static_cast<size_t>(bi) < objectss.size()) {
              batch_boxes[bi] = objectss[bi];
            }
          }
          y.reset();
        }

        const auto detect_end_steady = std::chrono::steady_clock::now();
        const long detect_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                detect_end_steady - detect_start_steady)
                .count();

        // ===========================
        // 第二次检测（铲斗 ROI 二次 YOLO）：
        // - 仅在 need_second_detection_ 打开且 second_yolo_ 存在时触发
        // - 输入仍然使用同一帧的 NV12 GPU buffer，不做 CPU 侧 cv::Mat
        // 裁剪/resize
        // - nv12_bucket_second_detection::Run 会：
        //   1) 在第一次检测框里找到 bucket/reset（取最可信的铲斗框）
        //   2) 若找到铲斗：在 GPU 上对 NV12 的 ROI 做 resize->BGR，写入
        //   second_yolo 输入 3) second_yolo 推理后把二次框坐标按 ROI
        //   映射回原图坐标，补充到输出框里
        //
        // 与第一次检测的主要不同点：
        // - 第一次检测是“整帧 ROI 区域”的目标检测（可能 batch>1）
        // - 第二次检测是“只对铲斗区域做细分类/细检测”，通常 batch=1
        // 且只跑在有铲斗时
        // ===========================
        // 写回每路的检测框 + 发布检测结果
        for (size_t bi = 0; bi < batch_sources.size(); ++bi) {
          const size_t sid = batch_sources[bi];
          auto &c = input_contexts_[sid];

          c.tooth_detection_config_.tooth_state_name.clear();
          c.tooth_detection_config_.teeth_root_coordinates.clear();
          c.tooth_detection_config_.root_idx.clear();
          c.pedestrian_detection_config_.people_cnt = 0;
          c.pedestrian_detection_config_.show_people_cnt = 0;

          const auto &names = yolo_->getParam().class_names;
          int people_idx = -1;
          for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == "people") {
              people_idx = static_cast<int>(i);
              break;
            }
          }

          std::vector<Box> people_boxes;
          std::vector<Box> non_people_boxes;
          int people_cnt = 0;

          const bool do_people_second_classify =
              (people_idx >= 0) && c.model_config_.need_second_detection_ &&
              c.model_config_.need_classifier_ && trt_classifier_ &&
              trt_classifier_->isReady();

          if (do_people_second_classify) {
            std::vector<Box> people_candidates;
            people_candidates.reserve(batch_boxes[bi].size());
            for (const auto &b : batch_boxes[bi]) {
              if (b.label == people_idx) {
                people_candidates.push_back(b);
              } else {
                non_people_boxes.push_back(b);
              }
            }

            people_boxes.reserve(people_candidates.size());
            for (auto &pb : people_candidates) {
              int x = static_cast<int>(pb.left);
              int y = static_cast<int>(pb.top);
              int w = static_cast<int>(pb.right - pb.left);
              int h = static_cast<int>(pb.bottom - pb.top);

              x = std::max(0, std::min(x, batch_frames[bi].width - 1));
              y = std::max(0, std::min(y, batch_frames[bi].height - 1));
              w = std::max(1, std::min(w, batch_frames[bi].width - x));
              h = std::max(1, std::min(h, batch_frames[bi].height - y));

              std::vector<float> classifier_output;
              {
                std::lock_guard<std::mutex> lock(trt_classifier_mutex_);
                trt_classifier_->predictFromGxfNv12GpuRoi(
                    batch_frames[bi].gpu_buffer, batch_frames[bi].width,
                    batch_frames[bi].height, batch_frames[bi].y_stride,
                    batch_frames[bi].uv_stride, x, y, w, h,
                    batch_frames[bi].ready_event, classifier_output);
              }

              if (!classifier_output.empty() && classifier_output[0] > 0.5f) {
                people_boxes.push_back(pb);
                people_cnt += 1;
              }
            }
          } else {
            constexpr float kPeopleConfidenceThreshold = 0.7f;
            SplitPrimaryBoxesByPeopleThreshold(
                batch_boxes[bi], people_idx, kPeopleConfidenceThreshold,
                people_boxes, non_people_boxes, people_cnt);
          }
          c.pedestrian_detection_config_.people_cnt = people_cnt;
          c.pedestrian_detection_config_.show_people_cnt = people_cnt;

          std::vector<Box> final_boxes;
          if (c.model_config_.need_second_detection_ && second_yolo_) {
            // 二次检测只针对铲斗
            // ROI：行人结果仅来自一次检测（阈值过滤后保留）
            std::vector<Box> bucket_and_teeth_boxes;
            {
              std::lock_guard<std::mutex> lock(global_detection_mutex_);
              nv12_bucket_second_detection::Run(
                  second_yolo_.get(), batch_frames[bi].gpu_buffer,
                  batch_frames[bi].width, batch_frames[bi].height,
                  batch_frames[bi].y_stride, batch_frames[bi].uv_stride,
                  batch_frames[bi].ready_event, non_people_boxes, names,
                  c.tooth_detection_config_, bucket_and_teeth_boxes);
            }
            final_boxes.reserve(people_boxes.size() +
                                bucket_and_teeth_boxes.size());
            final_boxes.insert(final_boxes.end(), people_boxes.begin(),
                               people_boxes.end());
            final_boxes.insert(final_boxes.end(),
                               bucket_and_teeth_boxes.begin(),
                               bucket_and_teeth_boxes.end());
          } else {
            // 不做二次检测：直接使用一次检测结果（对 people 做阈值过滤）
            final_boxes.reserve(people_boxes.size() + non_people_boxes.size());
            final_boxes.insert(final_boxes.end(), people_boxes.begin(),
                               people_boxes.end());
            final_boxes.insert(final_boxes.end(), non_people_boxes.begin(),
                               non_people_boxes.end());
            UpdateToothConfigFromPrimaryBoxes(final_boxes, names,
                                              c.tooth_detection_config_);
          }

          std::vector<std::vector<Box>> bboxes;
          bboxes.resize(1);
          bboxes[0] = std::move(final_boxes);

          // const long wait_ms =
          //     batch_has_enq_tps[bi]
          //         ? std::chrono::duration_cast<std::chrono::milliseconds>(
          //               detect_start_steady - batch_enq_tps[bi])
          //               .count()
          //         : -1;

          // const long total_ms = (wait_ms >= 0 ? wait_ms : 0) + detect_ms;
          // RCLCPP_INFO_THROTTLE(
          //     logger, *clock, 2000,
          //     "[DETECT] source=%zu detect=%ldms wait=%ldms total=%ldms", sid,
          //     detect_ms, wait_ms, total_ms);

          // // 发布检测结果（与旧 yolo_detect 路径一致的输出通道）
          // if (c.input_config_.ros2_enabled && ros2_client_ &&
          //     !bboxes[0].empty()) {
          //   const int cam_idx = c.input_config_.idx;
          //   for (const auto &box : bboxes[0]) {
          //     std::string cls = (box.label >= 0 &&
          //                        static_cast<size_t>(box.label) <
          //                        names.size())
          //                           ? names[box.label]
          //                           : std::string("unknown");
          //     ros2_client_->publishDetectionResult(box, cls, cam_idx);
          //   }
          // }

          // 只把结果框写回（其他线程读取它来绘制）
          {
            std::lock_guard<std::mutex> last_lock(c.last_detection_mutex);
            c.last_detection_boxes = std::move(bboxes);
          }
          {
            std::lock_guard<std::mutex> lock(c.image_processing_mutex);
            c.image_processing_config_.bboxes = c.last_detection_boxes;
          }

          HandlePeopleMqtt(c.tooth_detection_config_,
                           c.pedestrian_detection_config_, mqtt_publisher_,
                           mqtt_publisher_mutex_);
          const auto &tooth_class_names =
              (c.model_config_.need_second_detection_ && second_yolo_)
                  ? second_yolo_->getParam().class_names
                  : names;
          HandleToothMqtt(c.tooth_detection_config_, tooth_class_names,
                          mqtt_publisher_, mqtt_publisher_mutex_);
        }
      }

      // 释放“检测引用”（帧仍可能被发布线程使用）
      for (size_t bi = 0; bi < batch_sources.size(); ++bi) {
        const size_t sid = batch_sources[bi];
        auto &c = input_contexts_[sid];
        const auto &d = batch_frames[bi];
        if (d.pool_index >= 0) {
          bool push_gc = false;
          {
            std::lock_guard<std::mutex> pool_lock(c.nitros_nv12_pool_mutex_);
            if (static_cast<size_t>(d.pool_index) <
                c.nitros_nv12_pool_.size()) {
              auto &slot = c.nitros_nv12_pool_[d.pool_index];
              if (slot.ref_count > 0) {
                slot.ref_count -= 1;
              }
              if (slot.ref_count == 0) {
                push_gc = true;
              }
            }
          }
          if (push_gc) {
            std::lock_guard<std::mutex> gc_lock(c.nitros_nv12_pool_gc_mutex_);
            c.nitros_nv12_pool_gc_.push(d.pool_index);
          }
        } else {
          std::lock_guard<std::mutex> gc_lock(c.nitros_gpu_gc_mutex_);
          c.nitros_gpu_gc_queue_.push(d);
        }
      }

      current_source = (batch_sources.back() + 1) % num_sources;
    } else {
      // ========== 非 NITROS：保持原有单帧路径 ==========
      auto &ctx = input_contexts_[selected_source];
      std::unique_ptr<cv::Mat> undistorted_img;
      std::chrono::steady_clock::time_point enq_tp;
      bool has_enq_tp = false;
      {
        std::unique_lock<std::mutex> lock(ctx.detection_mutex_);
        // 只取最新帧，避免堆积导致延迟抖动
        while (ctx.detection_queue.size() > 1) {
          ctx.detection_queue.pop();
        }
        if (!ctx.detection_queue.empty()) {
          undistorted_img = std::move(ctx.detection_queue.front());
          ctx.detection_queue.pop();
        }
        has_enq_tp = ctx.has_last_detection_enqueue_tp_;
        enq_tp = ctx.last_detection_enqueue_tp_;
        ctx.has_last_detection_enqueue_tp_ = false;
      }

      if (!undistorted_img) {
        // 该源的 ready
        // 标志可能对应“新帧入队但又被最新帧覆盖/被取走”，直接继续
        current_source = (selected_source + 1) % num_sources;
        continue;
      }

      const auto detect_start_steady = std::chrono::steady_clock::now();

      const long wait_ms =
          has_enq_tp ? std::chrono::duration_cast<std::chrono::milliseconds>(
                           detect_start_steady - enq_tp)
                           .count()
                     : -1;

      // 检测使用本地 ImageProcessingConfig，避免长时间占用
      // ctx.image_processing_mutex
      ImageProcessingConfig local_cfg;
      {
        std::lock_guard<std::mutex> lock(ctx.image_processing_mutex);
        local_cfg = ctx.image_processing_config_;
      }
      local_cfg.init_frame = *undistorted_img;

      {
        std::lock_guard<std::mutex> global_lock(global_detection_mutex_);
        yolo_detect::detect(yolo_, second_yolo_, trt_classifier_,
                            ctx.undistorter_, mqtt_publisher_, ros2_client_,
                            ctx.model_config_, ctx.undistort_config_,
                            ctx.input_config_, ctx.tooth_detection_config_,
                            ctx.pedestrian_detection_config_, local_cfg,
                            general_config_, ctx.roi_config_, ctx.mqtt_config_);
      }

      // const auto detect_end_steady = std::chrono::steady_clock::now();
      // const long detect_ms =
      //     std::chrono::duration_cast<std::chrono::milliseconds>(
      //         detect_end_steady - detect_start_steady)
      //         .count();

      // const long total_ms = (wait_ms >= 0 ? wait_ms : 0) + detect_ms;
      // RCLCPP_INFO_THROTTLE(
      //     logger, *clock, 2000,
      //     "[DETECT] source=%zu detect=%ldms wait=%ldms total=%ldms",
      //     selected_source, detect_ms, wait_ms, total_ms);

      {
        std::lock_guard<std::mutex> last_lock(ctx.last_detection_mutex);
        ctx.last_detection_boxes = std::move(local_cfg.bboxes);
      }
      {
        std::lock_guard<std::mutex> lock(ctx.image_processing_mutex);
        ctx.image_processing_config_.bboxes = ctx.last_detection_boxes;
      }

      current_source = (selected_source + 1) % num_sources;
    }
  }

  sample::gLogInfo << "Detection thread exiting..." << std::endl;
}

void SharedModelMultiSourcesPipelineManager::start() {
  if (general_config_.show_gui && !use_nitros_sub_) {
    gui_thread_ = std::thread([this]() { GUI(); });
  }

  // 每路绘制/发布线程：
  // - 非 NV12：使用 CPU cv::Mat 叠框发布
  // - NV12：直接在 GPU NV12 上叠历史框并发布，保证帧率
  worker_threads_.reserve(num_sources);
  for (size_t i = 0; i < num_sources; ++i) {
    worker_threads_.emplace_back([this, i]() { processUndistortAndDraw(i); });
  }

  detection_thread_ = std::thread([this]() { processDetectionGlobal(); });

  if (input_config_.ros2_enabled) {
    if (!parent_node_) {
      // 独立模式：启动自己的执行器线程并等待
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      ros_executor_thread_ = std::thread([this]() {
        while (is_running_.load(std::memory_order_acquire)) {
          executor_->spin_some(std::chrono::milliseconds(100));
        }
        sample::gLogInfo << "ROS2 executor thread exiting..." << std::endl;
      });

      // 独立模式：等待执行器线程和所有工作线程结束
      if (ros_executor_thread_.joinable()) {
        ros_executor_thread_.join();
      }

      if (detection_thread_.joinable()) {
        detection_thread_.join();
      }

      for (auto &thread : worker_threads_) {
        if (thread.joinable())
          thread.join();
      }

      if (gui_thread_.joinable()) {
        gui_thread_.join();
      }
    } else {
      // 组件模式：不阻塞，线程在后台运行
      // 由组件容器的执行器驱动消息回调
      sample::gLogInfo << "Component mode: start() returns immediately, "
                       << "threads running in background." << std::endl;
    }
  } else {
    // 非 ROS2 模式：从本地文件/相机读取
    while (is_running_.load(std::memory_order_acquire)) {
      size_t empty_sources = 0;
      for (size_t i = 0; i < num_sources; ++i) {
        auto &ctx = input_contexts_[i];
        cv::Mat frame;
        ctx.image_processing_config_.captures[ctx.input_config_.idx].read(
            frame);
        if (frame.empty()) {
          if (++empty_sources == num_sources) {
            is_running_.store(false, std::memory_order_release);
            break;
          }
        } else {
          // V4L2 常输出 YUYV/YUY2（CV_8UC2）。后续推理/绘制按 BGR(3通道)
          // 使用会越界崩溃。
          if (frame.channels() != 3) {
            if (frame.type() == CV_8UC2) {
              cv::Mat bgr;
              try {
                cv::cvtColor(frame, bgr, cv::COLOR_YUV2BGR_YUY2);
              } catch (const cv::Exception &) {
                continue;
              }
              frame = std::move(bgr);
            } else if (frame.type() == CV_8UC4) {
              cv::Mat bgr;
              try {
                cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
              } catch (const cv::Exception &) {
                continue;
              }
              frame = std::move(bgr);
            } else {
              continue;
            }
          }
          enqueueRawFrame(static_cast<int>(i), std::move(frame));
        }
      }

      if (!is_running_.load(std::memory_order_acquire)) {
        sample::gLogInfo << "All input sources are empty. Exiting all threads."
                         << std::endl;
        is_running_.store(false, std::memory_order_release);
        break;
      }
    }

    // 等待所有线程结束
    for (auto &thread : decode_threads_) {
      if (thread.joinable())
        thread.join();
    }

    if (detection_thread_.joinable()) {
      detection_thread_.join();
    }

    for (auto &thread : worker_threads_) {
      if (thread.joinable())
        thread.join();
    }

    if (gui_thread_.joinable()) {
      gui_thread_.join();
    }
  }
}

void SharedModelMultiSourcesPipelineManager::stop() {
  is_running_.store(false, std::memory_order_release);

  for (auto &[id, ctx] : input_contexts_) {
    ctx.raw_frame_cond_.notify_all();
    ctx.detection_cond_.notify_all();
    ctx.display_cond_.notify_all();
    ctx.decode_cond_.notify_all();
    ctx.nitros_gpu_cond_.notify_all(); // 通知 GPU 队列
  }
  ready_cond_.notify_all();

  nitros_subscribers_.clear();
  image_raw_subscribers_.clear();

  if (input_config_.ros2_enabled && ros_executor_thread_.joinable() &&
      !parent_node_) {
    ros_executor_thread_.join();
  }

  for (auto &thread : decode_threads_) {
    if (thread.joinable())
      thread.join();
  }

  if (detection_thread_.joinable()) {
    detection_thread_.join();
  }

  for (auto &thread : worker_threads_) {
    if (thread.joinable())
      thread.join();
  }

  if (gui_thread_.joinable()) {
    gui_thread_.join();
  }

  cleanupResources();
}
