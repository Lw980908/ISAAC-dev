#include "single_thread_single_source.h"
#include "distortion_correction/undistort_factory.h"
#include "gxf_nv12_convert.h"
#include "gxf_nv12_draw.h"
#include "nv12_bucket_second_detection.h"
#include "yolo_detect.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

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

static std::string GetSingleCameraNamespace(const InputConfig &cfg) {
  if (!cfg.camera_namespaces.empty()) {
    return cfg.camera_namespaces[0];
  }
  if (!cfg.camera_ids.empty()) {
    return "camera" + std::to_string(cfg.camera_ids[0]);
  }
  return "camera0";
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

struct SingleThreadSingleSourcePipelineManager::Nv12PublishPool {
  explicit Nv12PublishPool(size_t bytes) : size_bytes(bytes) {}
  Nv12PublishPool(const Nv12PublishPool &) = delete;
  Nv12PublishPool &operator=(const Nv12PublishPool &) = delete;

  ~Nv12PublishPool() {
    std::lock_guard<std::mutex> lk(m);
    for (void *p : free_list) {
      if (p) {
        cudaFree(p);
      }
    }
    free_list.clear();
  }

  void *acquire() {
    std::lock_guard<std::mutex> lk(m);
    if (!free_list.empty()) {
      void *p = free_list.back();
      free_list.pop_back();
      return p;
    }
    void *p = nullptr;
    if (cudaMalloc(&p, size_bytes) != cudaSuccess) {
      return nullptr;
    }
    return p;
  }

  void warmup(size_t count) {
    std::lock_guard<std::mutex> lk(m);
    while (free_list.size() < count) {
      void *p = nullptr;
      if (cudaMalloc(&p, size_bytes) != cudaSuccess) {
        break;
      }
      free_list.push_back(p);
    }
  }

  void release(void *p) {
    if (!p) {
      return;
    }
    std::lock_guard<std::mutex> lk(m);
    free_list.push_back(p);
  }

  size_t size_bytes = 0;
  std::mutex m;
  std::vector<void *> free_list;
};

struct SingleThreadSingleSourcePipelineManager::Nv12SubPool {
  struct Slot {
    void *gpu_buffer{nullptr};
    size_t size_bytes{0};
    cudaEvent_t ready_event{nullptr};
    bool in_use{false};
    int ref_count{0};
  };

  explicit Nv12SubPool(size_t slots) {
    pool_.resize(std::max<size_t>(1, slots));
  }
  Nv12SubPool(const Nv12SubPool &) = delete;
  Nv12SubPool &operator=(const Nv12SubPool &) = delete;

  ~Nv12SubPool() {
    std::lock_guard<std::mutex> lk(m_);
    for (auto &s : pool_) {
      if (s.ready_event) {
        cudaEventDestroy(s.ready_event);
        s.ready_event = nullptr;
      }
      if (s.gpu_buffer) {
        cudaFree(s.gpu_buffer);
        s.gpu_buffer = nullptr;
      }
      s.size_bytes = 0;
      s.in_use = false;
    }
  }

  bool acquire(size_t bytes, int &pool_index, void *&gpu_buffer,
               cudaEvent_t &ready_event, int initial_ref_count) {
    if (initial_ref_count <= 0) {
      return false;
    }
    std::lock_guard<std::mutex> lk(m_);
    for (size_t i = 0; i < pool_.size(); ++i) {
      auto &s = pool_[i];
      if (s.in_use) {
        continue;
      }
      if (s.gpu_buffer && s.size_bytes != bytes) {
        cudaFree(s.gpu_buffer);
        s.gpu_buffer = nullptr;
        s.size_bytes = 0;
      }
      if (!s.gpu_buffer) {
        if (cudaMalloc(&s.gpu_buffer, bytes) != cudaSuccess) {
          return false;
        }
        s.size_bytes = bytes;
      }
      if (!s.ready_event) {
        if (cudaEventCreateWithFlags(&s.ready_event, cudaEventDisableTiming) !=
            cudaSuccess) {
          return false;
        }
      }
      s.in_use = true;
      s.ref_count = initial_ref_count;
      pool_index = static_cast<int>(i);
      gpu_buffer = s.gpu_buffer;
      ready_event = s.ready_event;
      return true;
    }
    return false;
  }

  void release(int pool_index) {
    if (pool_index < 0) {
      return;
    }
    std::lock_guard<std::mutex> lk(m_);
    if (static_cast<size_t>(pool_index) >= pool_.size()) {
      return;
    }
    auto &s = pool_[static_cast<size_t>(pool_index)];
    if (!s.in_use) {
      return;
    }
    if (s.ref_count > 0) {
      s.ref_count -= 1;
    }
    if (s.ref_count == 0) {
      s.in_use = false;
    }
  }

private:
  std::mutex m_;
  std::vector<Slot> pool_;
};

namespace {
static nvidia::isaac_ros::nitros::NitrosImage BuildNv12NitrosImageWithPool(
    const std_msgs::msg::Header &header, uint32_t height, uint32_t width,
    void *gpu_data, int y_stride, int uv_stride,
    const std::shared_ptr<
        SingleThreadSingleSourcePipelineManager::Nv12PublishPool> &pool) {
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

SingleThreadSingleSourcePipelineManager::
    SingleThreadSingleSourcePipelineManager()
    : SingleThreadSingleSourcePipelineManager(InputConfig{}) {}

SingleThreadSingleSourcePipelineManager::
    SingleThreadSingleSourcePipelineManager(const InputConfig &input_config) {
  this->input_config_ = input_config;
  if (!input_config_.use_ros2_component_container) {
    initResources();
  }
}

SingleThreadSingleSourcePipelineManager::
    ~SingleThreadSingleSourcePipelineManager() {
  stop();
}

bool SingleThreadSingleSourcePipelineManager::isRunning() const {
  return running_;
}

void SingleThreadSingleSourcePipelineManager::initResources() {
  if (!general_config_.is_init) {
    sample::gLogInfo << "Input source: "
                     << input_config_.input_source[static_cast<uint8_t>(
                            input_config_.source)]
                     << std::endl;
    yolo_ = std::make_unique<YOLOModel>(image_processing_config_, input_config_,
                                        model_config_, roi_config_);
    if (model_config_.need_second_detection_) {
      second_yolo_ = std::make_unique<YOLOModel>(
          image_processing_config_, input_config_, model_config_, roi_config_);
      // 初始化二次目标检测线程池
      model_config_.bucket_thread_pool_ = std::make_shared<ThreadPool>(
          model_config_.bucket_detection_thread_limit);
      sample::gLogInfo << "Second YOLO model initialized." << std::endl;
    }
    if (model_config_.need_classifier_) {
      // 初始化分类器线程池
      model_config_.classifier_thread_pool_ =
          std::make_shared<ThreadPool>(model_config_.classifier_thread_limit);
    }
    undistorter_ = createUndistorter(undistort_config_, input_config_);
    trt_classifier_ = std::make_unique<TensorRTInference>(model_config_);
    mqtt_publisher_ = std::make_unique<MQTTPublish>(mqtt_config_);

    // 初始化ROS2
    if (input_config_.ros2_enabled) {
      if (input_config_.use_ros2_component_container) {
        ros2_client_ =
            std::make_unique<ROS2Client>(parent_node_, input_config_, 1);
        sample::gLogInfo << "ROS2 initialized in component mode." << std::endl;
      } else {
        if (!rclcpp::ok()) {
          rclcpp::init(0, nullptr);
        }
        ros2_client_ =
            std::make_unique<ROS2Client>(input_config_, "yolo_ros2_node", 1);
        sample::gLogInfo << "ROS2 node initialized." << std::endl;
      }

      if (cuda_stream_ == nullptr) {
        cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
      }
      if (cuda_sub_stream_ == nullptr) {
        cudaStreamCreateWithFlags(&cuda_sub_stream_, cudaStreamNonBlocking);
      }
      if (cuda_publish_stream_ == nullptr) {
        cudaStreamCreateWithFlags(&cuda_publish_stream_, cudaStreamNonBlocking);
      }

      callback_group_ = ros2_client_->getNode()->create_callback_group(
          rclcpp::CallbackGroupType::MutuallyExclusive);

      rclcpp::QoS qos_profile(1);
      qos_profile.best_effort();
      qos_profile.keep_last(1);

      use_nitros_sub_ = (image_encoder_decoder_config_.encoding_format_sub ==
                         EncodingFormat::NITROS_NV12);

      if (use_nitros_sub_) {
        const std::string camera_ns = GetSingleCameraNamespace(input_config_);
        std::string topic_name = input_config_.nitros_topic_prefix.empty()
                                     ? ("/" + camera_ns + "/image_raw")
                                     : input_config_.nitros_topic_prefix + "/" +
                                           camera_ns + "/image_raw";

        if (!nitros_sub_pool_) {
          nitros_sub_pool_ = std::make_shared<Nv12SubPool>(3);
        }

        nitros_image_subscriber_ = std::make_shared<
            nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                nvidia::isaac_ros::nitros::NitrosImageView>>(
            ros2_client_->getNode().get(), topic_name,
            nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name,
            std::function<void(
                const nvidia::isaac_ros::nitros::NitrosImageView &)>(
                [this](const nvidia::isaac_ros::nitros::NitrosImageView &view) {
                  if (!running_) {
                    return;
                  }

                  const int width = view.GetWidth();
                  const int height = view.GetHeight();
                  const unsigned char *gpu_data = view.GetGpuData();
                  if (!gpu_data || width <= 0 || height <= 0) {
                    return;
                  }

                  const int y_stride = static_cast<int>(view.GetStride(0));
                  const int uv_stride = static_cast<int>(view.GetStride(1));
                  const size_t total_size =
                      static_cast<size_t>(view.GetSizeInBytes());
                  const int32_t stamp_sec = view.GetTimestampSeconds();
                  const uint32_t stamp_nanosec = view.GetTimestampNanoseconds();

                  if (!nitros_sub_pool_) {
                    return;
                  }
                  void *local_gpu_buffer = nullptr;
                  cudaEvent_t ready_event = nullptr;
                  int pool_index = -1;
                  if (!nitros_sub_pool_->acquire(total_size, pool_index,
                                                 local_gpu_buffer, ready_event,
                                                 2)) {
                    return;
                  }

                  cudaError_t err = cudaMemcpyAsync(
                      local_gpu_buffer, gpu_data, total_size,
                      cudaMemcpyDeviceToDevice, cuda_sub_stream_);
                  if (err != cudaSuccess) {
                    nitros_sub_pool_->release(pool_index);
                    return;
                  }
                  err = cudaEventRecord(ready_event, cuda_sub_stream_);
                  if (err != cudaSuccess) {
                    nitros_sub_pool_->release(pool_index);
                    return;
                  }

                  Nv12GpuFrame frame;
                  frame.gpu_buffer = local_gpu_buffer;
                  frame.width = width;
                  frame.height = height;
                  frame.y_stride = y_stride;
                  frame.uv_stride = uv_stride;
                  frame.size_bytes = total_size;
                  frame.ready_event = ready_event;
                  frame.pool_index = pool_index;
                  frame.stamp_sec = stamp_sec;
                  frame.stamp_nanosec = stamp_nanosec;

                  {
                    std::lock_guard<std::mutex> lock(nv12_queue_mutex_);
                    while (!nv12_queue_.empty()) {
                      auto old = nv12_queue_.front();
                      nv12_queue_.pop();
                      if (nitros_sub_pool_) {
                        nitros_sub_pool_->release(old.pool_index);
                      }
                    }
                    nv12_queue_.push(frame);
                  }
                  nv12_queue_cv_.notify_one();

                  {
                    DrawItem item;
                    item.is_nv12 = true;
                    item.nv12_frame = frame;
                    std::lock_guard<std::mutex> lk(draw_queue_mutex_);
                    while (draw_queue_.size() >= max_draw_queue_size_) {
                      DrawItem old = std::move(draw_queue_.front());
                      draw_queue_.pop();
                      if (old.is_nv12 && nitros_sub_pool_) {
                        nitros_sub_pool_->release(old.nv12_frame.pool_index);
                      }
                    }
                    draw_queue_.push(std::move(item));
                  }
                  draw_queue_cv_.notify_one();
                }),
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, qos_profile);
      } else {
        const std::string camera_ns = GetSingleCameraNamespace(input_config_);
        std::string topic_name = input_config_.nitros_topic_prefix.empty()
                                     ? ("/" + camera_ns + "/image_raw")
                                     : input_config_.nitros_topic_prefix + "/" +
                                           camera_ns + "/image_raw";
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = callback_group_;
        image_raw_subscriber_ =
            ros2_client_->getNode()
                ->create_subscription<sensor_msgs::msg::Image>(
                    topic_name, qos_profile,
                    [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                      if (!running_ || !msg) {
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
                      } else {
                        return;
                      }

                      if (bgr.empty()) {
                        return;
                      }

                      {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        if (image_queue_.size() >= max_queue_size_) {
                          image_queue_.pop();
                        }
                        image_queue_.push(std::move(bgr));
                      }
                      queue_cv_.notify_one();
                    },
                    sub_options);
      }

      const bool use_nitros_pub =
          (image_encoder_decoder_config_.encoding_format_pub ==
           EncodingFormat::NITROS_NV12);

      if (use_nitros_pub) {
        const std::string camera_ns = GetSingleCameraNamespace(input_config_);
        std::string pub_topic_name =
            input_config_.nitros_pub_topic_prefix.empty()
                ? ("/" + camera_ns + "/image_perceptual")
                : input_config_.nitros_pub_topic_prefix + "/" + camera_ns +
                      "/image_perceptual";
        nitros_image_publisher_ = std::make_shared<
            nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                nvidia::isaac_ros::nitros::NitrosImage>>(
            ros2_client_->getNode().get(), pub_topic_name,
            nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name,
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
            rclcpp::QoS(1));
      } else {
        const std::string camera_ns = GetSingleCameraNamespace(input_config_);
        image_raw_publisher_ =
            ros2_client_->getNode()->create_publisher<sensor_msgs::msg::Image>(
                "/" + camera_ns + "/image_perceptual/raw",
                rclcpp::QoS(1).reliable());
      }

      if (!input_config_.input_sizes.empty()) {
        const auto sz = input_config_.input_sizes[0];
        if (sz.width > 0 && sz.height > 0) {
          const size_t nv12_size =
              static_cast<size_t>(sz.width) * sz.height * 2;
          nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
          nitros_publish_pool_->warmup(6);
        }
      }
    } else {
      if (!openInputStream(image_processing_config_, input_config_)) {
        sample::gLogError << "Failed to open input stream." << std::endl;
        return;
      }
    }
    general_config_.is_init = true;
  }
}

void SingleThreadSingleSourcePipelineManager::cleanupResources() {
  // 智能指针显式资源释放
  yolo_.reset();
  if (model_config_.need_second_detection_) {
    second_yolo_.reset();
    // 释放二次目标检测线程池资源
    model_config_.bucket_thread_pool_.reset();
  }
  if (model_config_.need_classifier_) {
    // 释放分类器线程池资源
    model_config_.classifier_thread_pool_.reset();
  }
  trt_classifier_.reset();
  undistorter_.reset();
  mqtt_publisher_.reset();

  nitros_image_subscriber_.reset();
  nitros_image_publisher_.reset();
  image_raw_publisher_.reset();
  nitros_publish_pool_.reset();
  {
    std::lock_guard<std::mutex> lock(nv12_queue_mutex_);
    while (!nv12_queue_.empty()) {
      auto f = nv12_queue_.front();
      nv12_queue_.pop();
      if (nitros_sub_pool_) {
        nitros_sub_pool_->release(f.pool_index);
      }
    }
  }
  nitros_sub_pool_.reset();
  callback_group_.reset();

  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }
  if (cuda_sub_stream_) {
    cudaStreamDestroy(cuda_sub_stream_);
    cuda_sub_stream_ = nullptr;
  }
  if (cuda_publish_stream_) {
    cudaStreamDestroy(cuda_publish_stream_);
    cuda_publish_stream_ = nullptr;
  }
  if (cuda_nv12_sub_buffer_) {
    cudaFree(cuda_nv12_sub_buffer_);
    cuda_nv12_sub_buffer_ = nullptr;
    cuda_nv12_sub_buffer_size_ = 0;
  }
  if (cuda_bgr_sub_buffer_) {
    cudaFree(cuda_bgr_sub_buffer_);
    cuda_bgr_sub_buffer_ = nullptr;
    cuda_bgr_sub_buffer_size_ = 0;
  }
  if (cuda_bgr_publish_buffer_) {
    cudaFree(cuda_bgr_publish_buffer_);
    cuda_bgr_publish_buffer_ = nullptr;
    cuda_bgr_publish_buffer_size_ = 0;
  }
  if (cuda_bbox_buffer_) {
    cudaFree(cuda_bbox_buffer_);
    cuda_bbox_buffer_ = nullptr;
    cuda_bbox_buffer_size_ = 0;
  }

  // 释放视频捕获资源
  if (image_processing_config_.captures[input_config_.idx].isOpened()) {
    image_processing_config_.captures[input_config_.idx].release();
  }

  // 清理ROS2资源
  ros2_client_.reset();
  if (input_config_.ros2_enabled && rclcpp::ok() && !parent_node_) {
    rclcpp::shutdown();
  }

  sample::gLogInfo << "All threads have finished processing." << std::endl;
}

bool SingleThreadSingleSourcePipelineManager::publishNitrosImage(
    const cv::Mat &bgr_frame) {
  if (!input_config_.ros2_enabled || !ros2_client_ ||
      !nitros_image_publisher_ || bgr_frame.empty()) {
    return false;
  }
  if (cuda_stream_ == nullptr) {
    cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
  }

  const int width = bgr_frame.cols;
  const int height = bgr_frame.rows;
  const int y_stride = width;
  const int uv_stride = width * 2;
  const int bgr_stride = width * 3;

  const size_t bgr_size = static_cast<size_t>(bgr_stride) * height;
  auto logger = ros2_client_->getNode()->get_logger();
  if (!EnsureCudaBuffer(&cuda_bgr_publish_buffer_,
                        cuda_bgr_publish_buffer_size_, bgr_size, logger,
                        "publish bgr")) {
    return false;
  }

  const size_t nv12_size = static_cast<size_t>(width) * height * 2;
  if (!nitros_publish_pool_ || nitros_publish_pool_->size_bytes != nv12_size) {
    nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
    nitros_publish_pool_->warmup(2);
  }

  void *publish_buffer = nitros_publish_pool_->acquire();
  if (!publish_buffer) {
    return false;
  }

  cudaError_t cuda_err =
      cudaMemcpyAsync(cuda_bgr_publish_buffer_, bgr_frame.data, bgr_size,
                      cudaMemcpyHostToDevice, cuda_stream_);
  if (cuda_err != cudaSuccess) {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }

  cuda_err = ConvertBGRToGxfNV12(publish_buffer, cuda_bgr_publish_buffer_,
                                 width, height, y_stride, uv_stride, bgr_stride,
                                 cuda_stream_);
  if (cuda_err != cudaSuccess) {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }

  cuda_err = cudaStreamSynchronize(cuda_stream_);
  if (cuda_err != cudaSuccess) {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }

  std_msgs::msg::Header header;
  header.stamp = ros2_client_->getNode()->now();
  const int cam_id =
      input_config_.camera_ids.empty() ? 0 : input_config_.camera_ids[0];
  header.frame_id = "camera_" + std::to_string(cam_id);

  try {
    auto nitros_image = BuildNv12NitrosImageWithPool(
        header, static_cast<uint32_t>(height), static_cast<uint32_t>(width),
        publish_buffer, y_stride, uv_stride, nitros_publish_pool_);
    nitros_image_publisher_->publish(nitros_image);
  } catch (...) {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }
  return true;
}

bool SingleThreadSingleSourcePipelineManager::publishNitrosNv12WithOverlay(
    const Nv12GpuFrame &frame, const std::vector<std::vector<Box>> &bboxes) {
  if (!input_config_.ros2_enabled || !ros2_client_ ||
      !nitros_image_publisher_ || !frame.gpu_buffer || frame.width <= 0 ||
      frame.height <= 0 || frame.y_stride <= 0 || frame.uv_stride <= 0 ||
      frame.size_bytes == 0) {
    return false;
  }

  auto logger = ros2_client_->getNode()->get_logger();
  if (cuda_publish_stream_ == nullptr) {
    cudaStreamCreateWithFlags(&cuda_publish_stream_, cudaStreamNonBlocking);
  }

  const size_t nv12_size = frame.size_bytes;
  if (!nitros_publish_pool_ || nitros_publish_pool_->size_bytes != nv12_size) {
    nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
    nitros_publish_pool_->warmup(6);
  }
  auto pool = nitros_publish_pool_;

  void *publish_buffer = pool ? pool->acquire() : nullptr;
  if (!publish_buffer) {
    return false;
  }

  if (frame.ready_event) {
    cudaStreamWaitEvent(cuda_publish_stream_, frame.ready_event, 0);
  }
  cudaError_t perr =
      cudaMemcpyAsync(publish_buffer, frame.gpu_buffer, nv12_size,
                      cudaMemcpyDeviceToDevice, cuda_publish_stream_);

  std::vector<GxfBBox> people_bbs;
  std::vector<GxfCircleMark> marks;
  if (perr == cudaSuccess && !bboxes.empty() && !bboxes[0].empty()) {
    const auto &boxes0 = bboxes[0];
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
          (static_cast<double>(b.left) + static_cast<double>(b.right)) * 0.5));
      const int cy = static_cast<int>(std::lround(
          (static_cast<double>(b.top) + static_cast<double>(b.bottom)) * 0.5));

      const int x = std::max(0, std::min(cx, frame.width - 1));
      const int y = std::max(0, std::min(cy, frame.height - 1));

      marks.push_back({x, y, 5, yv, uv, vv});
    }
  }

  if (perr == cudaSuccess && !people_bbs.empty()) {
    if (EnsureCudaBuffer(&cuda_bbox_buffer_, cuda_bbox_buffer_size_,
                         people_bbs.size() * sizeof(GxfBBox), logger,
                         "bbox buffer")) {
      perr = cudaMemcpyAsync(cuda_bbox_buffer_, people_bbs.data(),
                             people_bbs.size() * sizeof(GxfBBox),
                             cudaMemcpyHostToDevice, cuda_publish_stream_);
      if (perr == cudaSuccess) {
        constexpr uint8_t y_val = 100;
        constexpr uint8_t u_val = 90;
        constexpr uint8_t v_val = 240;
        perr = DrawGxfNV12Bboxes(
            publish_buffer, frame.width, frame.height, frame.y_stride,
            frame.uv_stride, static_cast<const GxfBBox *>(cuda_bbox_buffer_),
            static_cast<int>(people_bbs.size()), 2, y_val, u_val, v_val,
            cuda_publish_stream_);
      }
    }
  }

  if (perr == cudaSuccess && !marks.empty()) {
    if (EnsureCudaBuffer(&cuda_bbox_buffer_, cuda_bbox_buffer_size_,
                         marks.size() * sizeof(GxfCircleMark), logger,
                         "circle marks buffer")) {
      perr = cudaMemcpyAsync(cuda_bbox_buffer_, marks.data(),
                             marks.size() * sizeof(GxfCircleMark),
                             cudaMemcpyHostToDevice, cuda_publish_stream_);
      if (perr == cudaSuccess) {
        perr = DrawGxfNV12CircleMarks(
            publish_buffer, frame.width, frame.height, frame.y_stride,
            frame.uv_stride,
            static_cast<const GxfCircleMark *>(cuda_bbox_buffer_),
            static_cast<int>(marks.size()), cuda_publish_stream_);
      }
    }
  }

  const int fps_x10 = static_cast<int>(std::lround(fps_value_ * 10.0));
  if (perr == cudaSuccess) {
    perr = DrawGxfNV12Fps(publish_buffer, frame.width, frame.height,
                          frame.y_stride, frame.uv_stride, fps_x10,
                          cuda_publish_stream_);
  }

  if (perr == cudaSuccess) {
    perr = cudaStreamSynchronize(cuda_publish_stream_);
  }
  if (perr != cudaSuccess) {
    if (pool) {
      pool->release(publish_buffer);
    } else {
      cudaFree(publish_buffer);
    }
    return false;
  }

  std_msgs::msg::Header header;
  header.stamp.sec = frame.stamp_sec;
  header.stamp.nanosec = frame.stamp_nanosec;
  const int cam_id =
      (input_config_.camera_ids.empty() ||
       static_cast<size_t>(input_config_.idx) >=
           input_config_.camera_ids.size())
          ? 0
          : input_config_.camera_ids[static_cast<size_t>(input_config_.idx)];
  header.frame_id = "camera_" + std::to_string(cam_id);

  try {
    auto nitros_image = BuildNv12NitrosImageWithPool(
        header, static_cast<uint32_t>(frame.height),
        static_cast<uint32_t>(frame.width), publish_buffer, frame.y_stride,
        frame.uv_stride, pool);
    nitros_image_publisher_->publish(nitros_image);
  } catch (...) {
    if (pool) {
      pool->release(publish_buffer);
    } else {
      cudaFree(publish_buffer);
    }
    return false;
  }
  return true;
}

void SingleThreadSingleSourcePipelineManager::drawThread() {
  while (running_) {
    DrawItem item;
    {
      std::unique_lock<std::mutex> lock(draw_queue_mutex_);
      draw_queue_cv_.wait(lock,
                          [this] { return !draw_queue_.empty() || !running_; });
      if (!running_ && draw_queue_.empty()) {
        return;
      }
      while (draw_queue_.size() > 1) {
        DrawItem old = std::move(draw_queue_.front());
        draw_queue_.pop();
        if (old.is_nv12 && nitros_sub_pool_) {
          nitros_sub_pool_->release(old.nv12_frame.pool_index);
        }
      }
      item = std::move(draw_queue_.front());
      draw_queue_.pop();
    }

    if (item.is_nv12) {
      if (input_config_.ros2_enabled && ros2_client_) {
        const auto now_tp = std::chrono::steady_clock::now();
        fps_window_frames_ += 1;
        const double sec =
            std::chrono::duration<double>(now_tp - fps_window_start_tp_)
                .count();
        if (sec >= 1.0) {
          fps_value_ = static_cast<double>(fps_window_frames_) / sec;
          fps_window_frames_ = 0;
          fps_window_start_tp_ = now_tp;
        }

        std::vector<std::vector<Box>> overlay_boxes;
        {
          std::lock_guard<std::mutex> last_lock(last_detection_mutex_);
          overlay_boxes = last_detection_boxes_;
        }

        if (image_encoder_decoder_config_.encoding_format_pub ==
            EncodingFormat::NITROS_NV12) {
          publishNitrosNv12WithOverlay(item.nv12_frame, overlay_boxes);
        } else if (image_raw_publisher_ || general_config_.show_gui) {
          if (cuda_stream_ == nullptr) {
            cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
          }
          cv::Mat bgr(item.nv12_frame.height, item.nv12_frame.width, CV_8UC3);
          const int bgr_stride = item.nv12_frame.width * 3;
          const size_t bgr_size =
              static_cast<size_t>(bgr_stride) * item.nv12_frame.height;
          auto logger = ros2_client_->getNode()->get_logger();
          if (EnsureCudaBuffer(&cuda_bgr_sub_buffer_, cuda_bgr_sub_buffer_size_,
                               bgr_size, logger, "bgr draw")) {
            cudaStreamWaitEvent(cuda_stream_, item.nv12_frame.ready_event);
            cudaError_t err = ConvertGxfNV12ToBGR(
                cuda_bgr_sub_buffer_, item.nv12_frame.gpu_buffer,
                item.nv12_frame.width, item.nv12_frame.height,
                item.nv12_frame.y_stride, item.nv12_frame.uv_stride, bgr_stride,
                cuda_stream_);
            if (err == cudaSuccess) {
              err = cudaMemcpyAsync(bgr.data, cuda_bgr_sub_buffer_, bgr_size,
                                    cudaMemcpyDeviceToHost, cuda_stream_);
              if (err == cudaSuccess) {
                cudaStreamSynchronize(cuda_stream_);
              }
            }
          }

          if (image_raw_publisher_ && !bgr.empty()) {
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr)
                           .toImageMsg();
            msg->header.stamp.sec = item.nv12_frame.stamp_sec;
            msg->header.stamp.nanosec = item.nv12_frame.stamp_nanosec;
            image_raw_publisher_->publish(*msg);
          }

          if (general_config_.show_gui && !bgr.empty()) {
            cv::resize(bgr, bgr, cv::Size(bgr.cols * 0.7, bgr.rows * 0.5));
            cv::imshow("Single-Source Display", bgr);
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27) {
              running_ = false;
              queue_cv_.notify_all();
              nv12_queue_cv_.notify_all();
              draw_queue_cv_.notify_all();
            }
          }
        }
      }

      if (nitros_sub_pool_) {
        nitros_sub_pool_->release(item.nv12_frame.pool_index);
      }
      continue;
    }

    ImageProcessingConfig proc;
    proc.init_frame = std::move(item.bgr_frame);
    proc.bboxes = std::move(item.bboxes);
    ToothDetectionConfig tooth = item.tooth_detection_config
                                     ? *item.tooth_detection_config
                                     : ToothDetectionConfig{};
    PedestrianDetectionConfig ped = item.pedestrian_detection_config
                                        ? *item.pedestrian_detection_config
                                        : PedestrianDetectionConfig{};
    InputConfig input = input_config_;
    ROIConfig roi = roi_config_;

    auto draw_start = std::chrono::high_resolution_clock::now();
    cv::Mat fusion_img = draw_detection_boxes(yolo_->getParam().class_names,
                                              proc, tooth, ped, input, roi);
    auto draw_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> draw_duration = draw_end - draw_start;
    sample::gLogInfo << "Draw time: " << draw_duration.count() * 1000 << "ms"
                     << std::endl;

    if (input_config_.ros2_enabled && ros2_client_) {
      if (image_encoder_decoder_config_.encoding_format_pub ==
          EncodingFormat::NITROS_NV12) {
        publishNitrosImage(fusion_img);
      } else if (image_raw_publisher_) {
        auto msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", fusion_img)
                .toImageMsg();
        msg->header.stamp = ros2_client_->getNode()->now();
        image_raw_publisher_->publish(*msg);
      }
    }

    if (general_config_.show_gui && !fusion_img.empty()) {
      cv::resize(fusion_img, fusion_img,
                 cv::Size(fusion_img.cols * 0.7, fusion_img.rows * 0.5));
      cv::imshow("Single-Source Display", fusion_img);
      char key = static_cast<char>(cv::waitKey(1));
      if (key == 27) {
        running_ = false;
        queue_cv_.notify_all();
        nv12_queue_cv_.notify_all();
        draw_queue_cv_.notify_all();
      }
    }
  }
}

void SingleThreadSingleSourcePipelineManager::detectionThread() {
  while (running_) {
    const bool skip_yolo_detect =
        (input_config_.ros2_enabled && use_nitros_sub_);
    if (input_config_.ros2_enabled) // ros2模式下读取相机图像
    {
      if (use_nitros_sub_) {
        Nv12GpuFrame nv12_frame;
        {
          std::unique_lock<std::mutex> lock(nv12_queue_mutex_);
          nv12_queue_cv_.wait(
              lock, [this] { return !nv12_queue_.empty() || !running_; });

          if (!running_ && nv12_queue_.empty()) {
            return;
          }

          while (nv12_queue_.size() > 1) {
            auto old = nv12_queue_.front();
            nv12_queue_.pop();
            if (nitros_sub_pool_) {
              nitros_sub_pool_->release(old.pool_index);
            }
          }
          nv12_frame = nv12_queue_.front();
          nv12_queue_.pop();
        }

        if (!nv12_frame.gpu_buffer || nv12_frame.width <= 0 ||
            nv12_frame.height <= 0) {
          if (nitros_sub_pool_) {
            nitros_sub_pool_->release(nv12_frame.pool_index);
          }
          continue;
        }

        std::vector<std::vector<Box>> bboxes;
        bboxes.resize(1);

        const bool detection_enabled =
            IsDetectionEnabledByMqtt(mqtt_publisher_, mqtt_publisher_mutex_);
        if (!detection_enabled) {
          ResetMqttAlarmsIfNeeded(tooth_detection_config_,
                                  pedestrian_detection_config_, mqtt_publisher_,
                                  mqtt_publisher_mutex_);

          tooth_detection_config_.tooth_state_name.clear();
          tooth_detection_config_.teeth_root_coordinates.clear();
          tooth_detection_config_.root_idx.clear();
          pedestrian_detection_config_.people_cnt = 0;
          pedestrian_detection_config_.show_people_cnt = 0;

          image_processing_config_.bboxes = bboxes;
        } else {
          tooth_detection_config_.tooth_state_name.clear();
          tooth_detection_config_.teeth_root_coordinates.clear();
          tooth_detection_config_.root_idx.clear();
          pedestrian_detection_config_.people_cnt = 0;
          pedestrian_detection_config_.show_people_cnt = 0;

          auto &y = yolo_->getYolo();
          y.copyFromGxfNv12Gpu(nv12_frame.gpu_buffer, nv12_frame.width,
                               nv12_frame.height, nv12_frame.y_stride,
                               nv12_frame.uv_stride, nv12_frame.ready_event, 0);
          y.preprocess();
          y.infer();
          y.postprocess(1);
          const auto objectss = y.getObjectss();
          y.reset();

          std::vector<Box> primary_boxes;
          if (!objectss.empty()) {
            primary_boxes = objectss[0];
          }

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
              (people_idx >= 0) && model_config_.need_second_detection_ &&
              model_config_.need_classifier_ && trt_classifier_ &&
              trt_classifier_->isReady();

          if (do_people_second_classify) {
            std::vector<Box> people_candidates;
            people_candidates.reserve(primary_boxes.size());
            for (const auto &b : primary_boxes) {
              if (b.label == people_idx) {
                people_candidates.push_back(b);
              } else {
                non_people_boxes.push_back(b);
              }
            }

            people_boxes.reserve(people_candidates.size());
            for (auto &pb : people_candidates) {
              int x = static_cast<int>(pb.left);
              int y0 = static_cast<int>(pb.top);
              int w = static_cast<int>(pb.right - pb.left);
              int h = static_cast<int>(pb.bottom - pb.top);

              x = std::max(0, std::min(x, nv12_frame.width - 1));
              y0 = std::max(0, std::min(y0, nv12_frame.height - 1));
              w = std::max(1, std::min(w, nv12_frame.width - x));
              h = std::max(1, std::min(h, nv12_frame.height - y0));

              std::vector<float> classifier_output;
              {
                std::lock_guard<std::mutex> lock(trt_classifier_mutex_);
                trt_classifier_->predictFromGxfNv12GpuRoi(
                    nv12_frame.gpu_buffer, nv12_frame.width, nv12_frame.height,
                    nv12_frame.y_stride, nv12_frame.uv_stride, x, y0, w, h,
                    nv12_frame.ready_event, classifier_output);
              }
              if (!classifier_output.empty() && classifier_output[0] > 0.5f) {
                people_boxes.push_back(pb);
                people_cnt += 1;
              }
            }
          } else {
            constexpr float kPeopleConfidenceThreshold = 0.7f;
            SplitPrimaryBoxesByPeopleThreshold(
                primary_boxes, people_idx, kPeopleConfidenceThreshold,
                people_boxes, non_people_boxes, people_cnt);
          }

          pedestrian_detection_config_.people_cnt = people_cnt;
          pedestrian_detection_config_.show_people_cnt = people_cnt;

          std::vector<Box> final_boxes;
          if (model_config_.need_second_detection_ && second_yolo_) {
            std::vector<Box> bucket_and_teeth_boxes;
            nv12_bucket_second_detection::Run(
                second_yolo_.get(), nv12_frame.gpu_buffer, nv12_frame.width,
                nv12_frame.height, nv12_frame.y_stride, nv12_frame.uv_stride,
                nv12_frame.ready_event, non_people_boxes, names,
                tooth_detection_config_, bucket_and_teeth_boxes);

            final_boxes.reserve(people_boxes.size() +
                                bucket_and_teeth_boxes.size());
            final_boxes.insert(final_boxes.end(), people_boxes.begin(),
                               people_boxes.end());
            final_boxes.insert(final_boxes.end(),
                               bucket_and_teeth_boxes.begin(),
                               bucket_and_teeth_boxes.end());
          } else {
            final_boxes.reserve(people_boxes.size() + non_people_boxes.size());
            final_boxes.insert(final_boxes.end(), people_boxes.begin(),
                               people_boxes.end());
            final_boxes.insert(final_boxes.end(), non_people_boxes.begin(),
                               non_people_boxes.end());
            UpdateToothConfigFromPrimaryBoxes(final_boxes, names,
                                              tooth_detection_config_);
          }

          bboxes[0] = std::move(final_boxes);
          image_processing_config_.bboxes = bboxes;

          HandlePeopleMqtt(tooth_detection_config_,
                           pedestrian_detection_config_, mqtt_publisher_,
                           mqtt_publisher_mutex_);
          const auto &tooth_class_names =
              (model_config_.need_second_detection_ && second_yolo_)
                  ? second_yolo_->getParam().class_names
                  : names;
          HandleToothMqtt(tooth_detection_config_, tooth_class_names,
                          mqtt_publisher_, mqtt_publisher_mutex_);
        }

        {
          std::lock_guard<std::mutex> last_lock(last_detection_mutex_);
          last_detection_boxes_ = image_processing_config_.bboxes;
        }

        if (nitros_sub_pool_) {
          nitros_sub_pool_->release(nv12_frame.pool_index);
        }
        continue;
      } else {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock,
                       [this] { return !image_queue_.empty() || !running_; });

        if (!running_ && image_queue_.empty())
          return;

        while (image_queue_.size() > 1) {
          image_queue_.pop();
        }
        image_processing_config_.init_frame =
            image_queue_.front(); // 只取最新的图像
        image_queue_.pop();
      }
    } else // 非ros2模式下读取帧
    {
      auto open_start = std::chrono::high_resolution_clock::now();
      image_processing_config_.captures[input_config_.idx].read(
          image_processing_config_.init_frame);
      if (image_processing_config_.init_frame.empty()) {
        running_ = false;
        break; // 立即退出当前线程的循环
      }
      // V4L2 常输出 YUYV/YUY2（CV_8UC2）。后续推理/绘制按 BGR(3通道)
      // 使用会越界崩溃。
      if (image_processing_config_.init_frame.channels() != 3) {
        if (image_processing_config_.init_frame.type() == CV_8UC2) {
          cv::Mat bgr;
          try {
            cv::cvtColor(image_processing_config_.init_frame, bgr,
                         cv::COLOR_YUV2BGR_YUY2);
          } catch (const cv::Exception &) {
            running_ = false;
            break;
          }
          image_processing_config_.init_frame = std::move(bgr);
        } else if (image_processing_config_.init_frame.type() == CV_8UC4) {
          cv::Mat bgr;
          try {
            cv::cvtColor(image_processing_config_.init_frame, bgr,
                         cv::COLOR_BGRA2BGR);
          } catch (const cv::Exception &) {
            running_ = false;
            break;
          }
          image_processing_config_.init_frame = std::move(bgr);
        } else {
          running_ = false;
          break;
        }
      }
      auto open_end = std::chrono::high_resolution_clock::now();
    }

    auto start_time = std::chrono::steady_clock::now();
    auto undistort_start = std::chrono::high_resolution_clock::now();
    cv::Mat undistort_img =
        undistort_config_.need_undistort_
            ? undistorter_->undistort_image(image_processing_config_.init_frame)
            : image_processing_config_.init_frame.clone();
    auto undistort_end = std::chrono::high_resolution_clock::now();
    auto undistort_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(undistort_end -
                                                              undistort_start);
    // std::cout << "Undistort time: " << undistort_duration.count() << "ms" <<
    // std::endl;

    image_processing_config_.init_frame = undistort_img.clone();

    if (!skip_yolo_detect) {
      auto detect_start = std::chrono::high_resolution_clock::now();
      if (model_config_.need_frame_skip_) {
        bool should_detect = process_frame_ % num_sources == 0;
        if (should_detect) {
          yolo_detect::detect(
              yolo_, second_yolo_, trt_classifier_, undistorter_,
              mqtt_publisher_, ros2_client_, model_config_, undistort_config_,
              input_config_, tooth_detection_config_,
              pedestrian_detection_config_, image_processing_config_,
              general_config_, roi_config_, mqtt_config_);

          previous_bboxes_ =
              image_processing_config_.bboxes; // 缓存当前检测结果
        } else {
          image_processing_config_.bboxes =
              previous_bboxes_; // 使用上一次的检测结果
        }
        process_frame_ = (process_frame_ + 1) % num_sources; // 更新处理帧计数
      } else {
        yolo_detect::detect(
            yolo_, second_yolo_, trt_classifier_, undistorter_, mqtt_publisher_,
            ros2_client_, model_config_, undistort_config_, input_config_,
            tooth_detection_config_, pedestrian_detection_config_,
            image_processing_config_, general_config_, roi_config_,
            mqtt_config_);
      }
      auto detect_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> detect_duration = detect_end - detect_start;
      sample::gLogInfo << "Detect time: " << detect_duration.count() * 1000
                       << " ms" << std::endl;
    }

    {
      DrawItem item;
      item.is_nv12 = false;
      item.bgr_frame = std::move(image_processing_config_.init_frame);
      item.bboxes = image_processing_config_.bboxes;
      item.tooth_detection_config =
          std::make_shared<ToothDetectionConfig>(tooth_detection_config_);
      item.pedestrian_detection_config =
          std::make_shared<PedestrianDetectionConfig>(
              pedestrian_detection_config_);

      std::lock_guard<std::mutex> lk(draw_queue_mutex_);
      while (draw_queue_.size() >= max_draw_queue_size_) {
        DrawItem old = std::move(draw_queue_.front());
        draw_queue_.pop();
        if (old.is_nv12 && nitros_sub_pool_) {
          nitros_sub_pool_->release(old.nv12_frame.pool_index);
        }
      }
      draw_queue_.push(std::move(item));
    }
    draw_queue_cv_.notify_one();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    sample::gLogInfo << "Total time = " << (1000 * time_used.count()) << " ms"
                     << std::endl;
    sample::gLogInfo << "---------------------------------------------------"
                     << std::endl;
  }
}

void SingleThreadSingleSourcePipelineManager::start() {
  // 启动检测线程
  detection_thread_ = std::thread(
      &SingleThreadSingleSourcePipelineManager::detectionThread, this);
  draw_thread_ =
      std::thread(&SingleThreadSingleSourcePipelineManager::drawThread, this);

  if (input_config_.ros2_enabled) {
    if (parent_node_) {
      return;
    }
    ros_spin_thread_ = std::thread([this]() {
      while (running_) {
        rclcpp::spin_some(ros2_client_->get_node_base_interface());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }

  if (detection_thread_.joinable()) {
    detection_thread_.join();
  }
  if (draw_thread_.joinable()) {
    draw_thread_.join();
  }
  running_ = false;
  queue_cv_.notify_all();
  nv12_queue_cv_.notify_all();
  draw_queue_cv_.notify_all();
  if (ros_spin_thread_.joinable()) {
    ros_spin_thread_.join();
  }
}

void SingleThreadSingleSourcePipelineManager::stop() {
  running_ = false;
  queue_cv_.notify_all();
  nv12_queue_cv_.notify_all();
  draw_queue_cv_.notify_all();
  if (detection_thread_.joinable()) {
    detection_thread_.join();
  }
  if (draw_thread_.joinable()) {
    draw_thread_.join();
  }
  if (ros_spin_thread_.joinable()) {
    ros_spin_thread_.join();
  }
  cleanupResources();
}
