#include "camera_combined_with_lidar.h"
#include "distortion_correction/undistort_factory.h"
#include "gxf_nv12_convert.h"
#include "gxf_nv12_draw.h"
#include "nv12_bucket_second_detection.h"
#include "yolo_detect.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
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

namespace
{
  constexpr uint64_t kNsPerSecLocal = 1000000000ULL;

  uint8_t ClipU8(int v)
  {
    if (v < 0)
    {
      return 0;
    }
    if (v > 255)
    {
      return 255;
    }
    return static_cast<uint8_t>(v);
  }

  cv::Scalar BgrFromYuv(uint8_t y, uint8_t u, uint8_t v)
  {
    const int c = static_cast<int>(y) - 16;
    const int d = static_cast<int>(u) - 128;
    const int e = static_cast<int>(v) - 128;
    const int r = (298 * c + 409 * e + 128) >> 8;
    const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    const int b = (298 * c + 516 * d + 128) >> 8;
    return cv::Scalar(ClipU8(b), ClipU8(g), ClipU8(r));
  }

  void DrawDashedHLine(cv::Mat &img, int x0, int x1, int y, const cv::Scalar &c,
                       int thickness, int dash_len, int gap_len)
  {
    if (img.empty() || y < 0 || y >= img.rows)
    {
      return;
    }
    if (x0 > x1)
    {
      std::swap(x0, x1);
    }
    x0 = std::max(0, std::min(x0, img.cols - 1));
    x1 = std::max(0, std::min(x1, img.cols - 1));
    for (int x = x0; x <= x1; x += (dash_len + gap_len))
    {
      const int xe = std::min(x + dash_len, x1);
      cv::line(img, cv::Point(x, y), cv::Point(xe, y), c, thickness, cv::LINE_AA);
    }
  }

  void DrawDashedVLine(cv::Mat &img, int x, int y0, int y1, const cv::Scalar &c,
                       int thickness, int dash_len, int gap_len)
  {
    if (img.empty() || x < 0 || x >= img.cols)
    {
      return;
    }
    if (y0 > y1)
    {
      std::swap(y0, y1);
    }
    y0 = std::max(0, std::min(y0, img.rows - 1));
    y1 = std::max(0, std::min(y1, img.rows - 1));
    for (int y = y0; y <= y1; y += (dash_len + gap_len))
    {
      const int ye = std::min(y + dash_len, y1);
      cv::line(img, cv::Point(x, y), cv::Point(x, ye), c, thickness, cv::LINE_AA);
    }
  }

  bool EnsureCudaBuffer(void **buffer, size_t &buffer_size, size_t required_size,
                        rclcpp::Logger logger, const char *tag)
  {
    if (*buffer && buffer_size >= required_size)
    {
      return true;
    }
    if (*buffer)
    {
      cudaFree(*buffer);
      *buffer = nullptr;
      buffer_size = 0;
    }
    cudaError_t err = cudaMalloc(buffer, required_size);
    if (err != cudaSuccess)
    {
      RCLCPP_ERROR(logger, "Failed to allocate CUDA buffer for %s: %s", tag,
                   cudaGetErrorString(err));
      return false;
    }
    buffer_size = required_size;
    return true;
  }

  static std::string GetSingleCameraNamespace(const InputConfig &cfg)
  {
    if (!cfg.camera_namespaces.empty())
    {
      return cfg.camera_namespaces[0];
    }
    if (!cfg.camera_ids.empty())
    {
      return "camera" + std::to_string(cfg.camera_ids[0]);
    }
    return "camera0";
  }

  void SplitPrimaryBoxesByPeopleThreshold(const std::vector<Box> &primary_boxes,
                                          int people_label, float threshold,
                                          std::vector<Box> &people_boxes,
                                          std::vector<Box> &non_people_boxes,
                                          int &people_cnt)
  {
    people_boxes.clear();
    non_people_boxes.clear();
    people_cnt = 0;
    for (const auto &b : primary_boxes)
    {
      if (people_label >= 0 && b.label == people_label)
      {
        if (b.confidence >= threshold)
        {
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
      ToothDetectionConfig &tooth_detection_config)
  {
    for (const auto &box : boxes)
    {
      if (box.label < 0 || static_cast<size_t>(box.label) >= class_names.size())
      {
        continue;
      }
      const auto &cls = class_names[static_cast<size_t>(box.label)];
      if (tooth_detection_config.tooth_state.count(box.label))
      {
        tooth_detection_config.tooth_state_name.emplace(cls);
        if (tooth_detection_config.excavator_type ==
            ExcavatorType::Front_shovel_excavator)
        {
          tooth_detection_config.teeth_root_coordinates.emplace_back(
              std::make_tuple(box.label, static_cast<int>(box.left - 50),
                              static_cast<int>(box.left - 50)));
          tooth_detection_config.teeth_root_coordinates.emplace_back(
              std::make_tuple(box.label, static_cast<int>(box.right + 50),
                              static_cast<int>(box.right + 50)));
        }
      }
      else if (tooth_detection_config.teeth_and_root.count(box.label))
      {
        tooth_detection_config.teeth_root_coordinates.emplace_back(
            std::make_tuple(box.label, static_cast<int>(box.left),
                            static_cast<int>(box.right)));
      }
    }
  }

  bool IsDetectionEnabledByMqtt(std::unique_ptr<MQTTPublish> &mqtt_publisher,
                                std::mutex &mqtt_mtx)
  {
    std::lock_guard<std::mutex> lock(mqtt_mtx);
    if (!mqtt_publisher)
    {
      return true;
    }
    return mqtt_publisher->is_subscribe_enabled();
  }

  bool PublishMqttJson(std::unique_ptr<MQTTPublish> &mqtt_publisher,
                       std::mutex &mqtt_mtx, const nlohmann::json &payload_json)
  {
    std::lock_guard<std::mutex> lock(mqtt_mtx);
    if (!mqtt_publisher || !mqtt_publisher->isMqttConnected())
    {
      return false;
    }
    return mqtt_publisher->publishAsync(payload_json.dump());
  }

  void ResetMqttAlarmsIfNeeded(
      ToothDetectionConfig &tooth_detection_config,
      PedestrianDetectionConfig &pedestrian_detection_config,
      std::unique_ptr<MQTTPublish> &mqtt_publisher, std::mutex &mqtt_mtx)
  {
    if (pedestrian_detection_config.is_people_detected)
    {
      nlohmann::json payload_json;
      payload_json["peopleCount"] = 0;
      if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json))
      {
        pedestrian_detection_config.last_people_detected_time_ =
            std::chrono::steady_clock::time_point();
        pedestrian_detection_config.is_people_detected = false;
        pedestrian_detection_config.show_people_cnt = 0;
      }
    }

    if (tooth_detection_config.is_tooth_detected)
    {
      nlohmann::json payload_json;
      payload_json["toothDetection"] =
          tooth_detection_config.detection_mapping["complete"];
      if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json))
      {
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
                        std::mutex &mqtt_mtx)
  {
    (void)tooth_detection_config;
    pedestrian_detection_config.show_people_cnt =
        pedestrian_detection_config.people_cnt;
    if (pedestrian_detection_config.show_people_cnt > 0)
    {
      nlohmann::json payload_json;
      payload_json["peopleCount"] = pedestrian_detection_config.show_people_cnt;
      if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json))
      {
        pedestrian_detection_config.last_people_detected_time_ =
            std::chrono::steady_clock::now();
        pedestrian_detection_config.is_people_detected = true;
      }
    }

    if (pedestrian_detection_config.is_people_detected)
    {
      const auto current_time = std::chrono::steady_clock::now();
      if (pedestrian_detection_config.last_people_detected_time_ !=
              std::chrono::steady_clock::time_point() &&
          (current_time -
           pedestrian_detection_config.last_people_detected_time_) >
              std::chrono::seconds(1))
      {
        if (pedestrian_detection_config.show_people_cnt == 0)
        {
          nlohmann::json payload_json;
          payload_json["peopleCount"] = 0;
          if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json))
          {
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
                       std::mutex &mqtt_mtx)
  {
    if (tooth_detection_config.tooth_state_name.size() == 1)
    {
      const std::string state = *tooth_detection_config.tooth_state_name.begin();
      if (tooth_detection_config.excavator_type ==
          ExcavatorType::Back_shovel_excavator)
      {
        if (static_cast<int>(
                tooth_detection_config.teeth_root_coordinates.size()) ==
            tooth_detection_config.teeth_complete_number)
        {
          std::sort(tooth_detection_config.teeth_root_coordinates.begin(),
                    tooth_detection_config.teeth_root_coordinates.end(),
                    [&](const std::tuple<int, int, int> &a,
                        const std::tuple<int, int, int> &b)
                    {
                      return std::get<1>(a) < std::get<1>(b);
                    });
          tooth_detection_config.root_idx.clear();
          for (int i = 0;
               i < static_cast<int>(
                       tooth_detection_config.teeth_root_coordinates.size());
               ++i)
          {
            const int lbl =
                std::get<0>(tooth_detection_config.teeth_root_coordinates[i]);
            if (lbl >= 0 &&
                static_cast<size_t>(lbl) < class_names_for_root.size() &&
                class_names_for_root[static_cast<size_t>(lbl)] == "root")
            {
              tooth_detection_config.root_idx.emplace_back(i + 1);
            }
          }

          if (tooth_detection_config.root_idx.empty() && state == "complete")
          {
            ++tooth_detection_config.detect_res["complete"];
            ++tooth_detection_config.match_cnt;
            if (tooth_detection_config.detect_res["complete"] >
                tooth_detection_config.max_count)
            {
              tooth_detection_config.max_count =
                  tooth_detection_config.detect_res["complete"];
              tooth_detection_config.max_class_label = "complete";
            }
          }
          else
          {
            for (const auto &idx : tooth_detection_config.root_idx)
            {
              const std::string label = "miss" + std::to_string(idx);
              if (state == label)
              {
                ++tooth_detection_config.detect_res[label];
                ++tooth_detection_config.match_cnt;
                if (tooth_detection_config.detect_res[label] >
                    tooth_detection_config.max_count)
                {
                  tooth_detection_config.max_count =
                      tooth_detection_config.detect_res[label];
                  tooth_detection_config.max_class_label = label;
                }
              }
            }
          }
        }
      }
      else
      {
        if (static_cast<int>(
                tooth_detection_config.teeth_root_coordinates.size()) ==
                tooth_detection_config.teeth_complete_number + 2 &&
            state == "complete")
        {
          ++tooth_detection_config.detect_res["complete"];
          ++tooth_detection_config.match_cnt;
          if (tooth_detection_config.detect_res["complete"] >
              tooth_detection_config.max_count)
          {
            tooth_detection_config.max_count =
                tooth_detection_config.detect_res["complete"];
            tooth_detection_config.max_class_label = "complete";
          }
        }
        else
        {
          std::sort(tooth_detection_config.teeth_root_coordinates.begin(),
                    tooth_detection_config.teeth_root_coordinates.end(),
                    [&](const std::tuple<int, int, int> &a,
                        const std::tuple<int, int, int> &b)
                    {
                      return std::get<1>(a) < std::get<1>(b);
                    });
          tooth_detection_config.root_idx.clear();
          for (int i = 0;
               i + 1 < static_cast<int>(
                           tooth_detection_config.teeth_root_coordinates.size());
               ++i)
          {
            const int dist = std::abs(
                std::get<1>(
                    tooth_detection_config.teeth_root_coordinates[i + 1]) -
                std::get<1>(tooth_detection_config.teeth_root_coordinates[i]));
            tooth_detection_config.root_idx.emplace_back(dist);
          }
          if (!tooth_detection_config.root_idx.empty())
          {
            const auto max_it =
                std::max_element(tooth_detection_config.root_idx.begin(),
                                 tooth_detection_config.root_idx.end());
            const auto max_gap_idx = static_cast<int>(
                max_it - tooth_detection_config.root_idx.begin());
            const std::string label = "miss" + std::to_string(max_gap_idx + 1);
            if (state == label)
            {
              ++tooth_detection_config.detect_res[label];
              ++tooth_detection_config.match_cnt;
              if (tooth_detection_config.detect_res[label] >
                  tooth_detection_config.max_count)
              {
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
              static_cast<int>(tooth_detection_config.match_cnt * 0.7))
      {
        tooth_detection_config.show_tooth_state_name =
            tooth_detection_config.max_class_label;
        if (tooth_detection_config.max_class_label != "complete")
        {
          nlohmann::json payload_json;
          payload_json["toothDetection"] =
              tooth_detection_config
                  .detection_mapping[tooth_detection_config.max_class_label];
          if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json))
          {
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
    }
    else
    {
      tooth_detection_config.show_tooth_state_name.clear();
    }

    if (tooth_detection_config.is_tooth_detected)
    {
      const auto current_time = std::chrono::steady_clock::now();
      if (tooth_detection_config.last_tooth_detected_time_ !=
              std::chrono::steady_clock::time_point() &&
          (current_time - tooth_detection_config.last_tooth_detected_time_) >
              std::chrono::seconds(1))
      {
        if (tooth_detection_config.teeth_root_coordinates.empty())
        {
          nlohmann::json payload_json;
          payload_json["toothDetection"] =
              tooth_detection_config.detection_mapping["complete"];
          if (PublishMqttJson(mqtt_publisher, mqtt_mtx, payload_json))
          {
            tooth_detection_config.last_tooth_detected_time_ =
                std::chrono::steady_clock::time_point();
            tooth_detection_config.is_tooth_detected = false;
          }
        }
      }
    }
  }
} // namespace

struct CameraCombinedWithLidarPipelineManager::Nv12PublishPool
{
  explicit Nv12PublishPool(size_t bytes) : size_bytes(bytes) {}
  Nv12PublishPool(const Nv12PublishPool &) = delete;
  Nv12PublishPool &operator=(const Nv12PublishPool &) = delete;

  ~Nv12PublishPool()
  {
    std::lock_guard<std::mutex> lk(m);
    for (void *p : free_list)
    {
      if (p)
      {
        cudaFree(p);
      }
    }
    free_list.clear();
  }

  void *acquire()
  {
    std::lock_guard<std::mutex> lk(m);
    if (!free_list.empty())
    {
      void *p = free_list.back();
      free_list.pop_back();
      return p;
    }
    void *p = nullptr;
    if (cudaMalloc(&p, size_bytes) != cudaSuccess)
    {
      return nullptr;
    }
    return p;
  }

  void warmup(size_t count)
  {
    std::lock_guard<std::mutex> lk(m);
    while (free_list.size() < count)
    {
      void *p = nullptr;
      if (cudaMalloc(&p, size_bytes) != cudaSuccess)
      {
        break;
      }
      free_list.push_back(p);
    }
  }

  void release(void *p)
  {
    if (!p)
    {
      return;
    }
    std::lock_guard<std::mutex> lk(m);
    free_list.push_back(p);
  }

  size_t size_bytes = 0;
  std::mutex m;
  std::vector<void *> free_list;
};

struct CameraCombinedWithLidarPipelineManager::Nv12SubPool
{
  struct Slot
  {
    void *gpu_buffer{nullptr};
    size_t size_bytes{0};
    cudaEvent_t ready_event{nullptr};
    bool in_use{false};
    int ref_count{0};
  };

  explicit Nv12SubPool(size_t slots)
  {
    pool_.resize(std::max<size_t>(1, slots));
  }
  Nv12SubPool(const Nv12SubPool &) = delete;
  Nv12SubPool &operator=(const Nv12SubPool &) = delete;

  ~Nv12SubPool()
  {
    std::lock_guard<std::mutex> lk(m_);
    for (auto &s : pool_)
    {
      if (s.ready_event)
      {
        cudaEventDestroy(s.ready_event);
        s.ready_event = nullptr;
      }
      if (s.gpu_buffer)
      {
        cudaFree(s.gpu_buffer);
        s.gpu_buffer = nullptr;
      }
      s.size_bytes = 0;
      s.in_use = false;
    }
  }

  bool acquire(size_t bytes, int &pool_index, void *&gpu_buffer,
               cudaEvent_t &ready_event, int initial_ref_count)
  {
    if (initial_ref_count <= 0)
    {
      return false;
    }
    std::lock_guard<std::mutex> lk(m_);
    for (size_t i = 0; i < pool_.size(); ++i)
    {
      auto &s = pool_[i];
      if (s.in_use)
      {
        continue;
      }
      if (s.gpu_buffer && s.size_bytes != bytes)
      {
        cudaFree(s.gpu_buffer);
        s.gpu_buffer = nullptr;
        s.size_bytes = 0;
      }
      if (!s.gpu_buffer)
      {
        if (cudaMalloc(&s.gpu_buffer, bytes) != cudaSuccess)
        {
          return false;
        }
        s.size_bytes = bytes;
      }
      if (!s.ready_event)
      {
        if (cudaEventCreateWithFlags(&s.ready_event, cudaEventDisableTiming) !=
            cudaSuccess)
        {
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

  void release(int pool_index)
  {
    if (pool_index < 0)
    {
      return;
    }
    std::lock_guard<std::mutex> lk(m_);
    if (static_cast<size_t>(pool_index) >= pool_.size())
    {
      return;
    }
    auto &s = pool_[static_cast<size_t>(pool_index)];
    if (!s.in_use)
    {
      return;
    }
    if (s.ref_count > 0)
    {
      s.ref_count -= 1;
    }
    if (s.ref_count == 0)
    {
      s.in_use = false;
    }
  }

private:
  std::mutex m_;
  std::vector<Slot> pool_;
};

namespace
{
  static nvidia::isaac_ros::nitros::NitrosImage BuildNv12NitrosImageWithPool(
      const std_msgs::msg::Header &header, uint32_t height, uint32_t width,
      void *gpu_data, int y_stride, int uv_stride,
      const std::shared_ptr<
          CameraCombinedWithLidarPipelineManager::Nv12PublishPool> &pool)
  {
    auto message = nvidia::gxf::Entity::New(
        nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext());
    if (!message)
    {
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
    if (!output_timestamp)
    {
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
    if (!gxf_image)
    {
      std::stringstream error_msg;
      error_msg << "[BuildNv12NitrosImageWithPool] Failed to add VideoBuffer: "
                << GxfResultStr(gxf_image.error());
      throw std::runtime_error(error_msg.str().c_str());
    }

    if (width % 2 != 0 || height % 2 != 0)
    {
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
                                  [pool](void *ptr)
                                  {
                                    if (pool)
                                    {
                                      pool->release(ptr);
                                      return nvidia::gxf::Success;
                                    }
                                    cudaFree(ptr);
                                    return nvidia::gxf::Success;
                                  });

    return nitros_image;
  }
} // namespace

CameraCombinedWithLidarPipelineManager::CameraCombinedWithLidarPipelineManager()
    : CameraCombinedWithLidarPipelineManager(InputConfig{}) {}

CameraCombinedWithLidarPipelineManager::CameraCombinedWithLidarPipelineManager(
    const InputConfig &input_config)
{
  this->input_config_ = input_config;
  if (!input_config_.use_ros2_component_container)
  {
    initResources();
  }
}

CameraCombinedWithLidarPipelineManager::
    ~CameraCombinedWithLidarPipelineManager()
{
  stop();
}

bool CameraCombinedWithLidarPipelineManager::isRunning() const
{
  return running_;
}

void CameraCombinedWithLidarPipelineManager::initResources()
{
  if (!general_config_.is_init)
  {
    sample::gLogInfo << "Input source: "
                     << input_config_.input_source[static_cast<uint8_t>(
                            input_config_.source)]
                     << std::endl;
    yolo_ = std::make_unique<YOLOModel>(image_processing_config_, input_config_,
                                        model_config_, roi_config_);
    if (model_config_.need_second_detection_)
    {
      second_yolo_ = std::make_unique<YOLOModel>(
          image_processing_config_, input_config_, model_config_, roi_config_);
      // 初始化二次目标检测线程池
      model_config_.bucket_thread_pool_ = std::make_shared<ThreadPool>(
          model_config_.bucket_detection_thread_limit);
      sample::gLogInfo << "Second YOLO model initialized." << std::endl;
    }
    if (model_config_.need_classifier_)
    {
      // 初始化分类器线程池
      model_config_.classifier_thread_pool_ =
          std::make_shared<ThreadPool>(model_config_.classifier_thread_limit);
    }
    undistorter_ = createUndistorter(undistort_config_, input_config_);
    trt_classifier_ = std::make_unique<TensorRTInference>(model_config_);
    mqtt_publisher_ = std::make_unique<MQTTPublish>(mqtt_config_);

    // 初始化ROS2
    if (input_config_.ros2_enabled)
    {
      if (input_config_.use_ros2_component_container)
      {
        ros2_client_ =
            std::make_unique<ROS2Client>(parent_node_, input_config_, 1);
        sample::gLogInfo << "ROS2 initialized in component mode." << std::endl;
      }
      else
      {
        if (!rclcpp::ok())
        {
          rclcpp::init(0, nullptr);
        }
        ros2_client_ =
            std::make_unique<ROS2Client>(input_config_, "yolo_ros2_node", 1);
        sample::gLogInfo << "ROS2 node initialized." << std::endl;
      }

      if (cuda_stream_ == nullptr)
      {
        cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
      }
      if (cuda_sub_stream_ == nullptr)
      {
        cudaStreamCreateWithFlags(&cuda_sub_stream_, cudaStreamNonBlocking);
      }
      if (cuda_publish_stream_ == nullptr)
      {
        cudaStreamCreateWithFlags(&cuda_publish_stream_, cudaStreamNonBlocking);
      }

      callback_group_ = ros2_client_->getNode()->create_callback_group(
          rclcpp::CallbackGroupType::MutuallyExclusive);

      rclcpp::QoS qos_profile(1);
      qos_profile.best_effort();
      qos_profile.keep_last(1);

      use_nitros_sub_ = (image_encoder_decoder_config_.encoding_format_sub ==
                         EncodingFormat::NITROS_NV12);

      if (use_nitros_sub_)
      {
        const std::string camera_ns = GetSingleCameraNamespace(input_config_);
        std::string topic_name = input_config_.nitros_topic_prefix.empty()
                                     ? ("/" + camera_ns + "/image_raw")
                                     : input_config_.nitros_topic_prefix + "/" +
                                           camera_ns + "/image_raw";

        if (!nitros_sub_pool_)
        {
          nitros_sub_pool_ = std::make_shared<Nv12SubPool>(3);
        }

        nitros_image_subscriber_ = std::make_shared<
            nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                nvidia::isaac_ros::nitros::NitrosImageView>>(
            ros2_client_->getNode().get(), topic_name,
            nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name,
            std::function<void(
                const nvidia::isaac_ros::nitros::NitrosImageView &)>(
                [this](const nvidia::isaac_ros::nitros::NitrosImageView &view)
                {
                  if (!running_)
                  {
                    return;
                  }

                  const int width = view.GetWidth();
                  const int height = view.GetHeight();
                  const unsigned char *gpu_data = view.GetGpuData();
                  if (!gpu_data || width <= 0 || height <= 0)
                  {
                    return;
                  }

                  const int y_stride = static_cast<int>(view.GetStride(0));
                  const int uv_stride = static_cast<int>(view.GetStride(1));
                  const size_t total_size =
                      static_cast<size_t>(view.GetSizeInBytes());
                  const int32_t stamp_sec = view.GetTimestampSeconds();
                  const uint32_t stamp_nanosec = view.GetTimestampNanoseconds();

                  if (!nitros_sub_pool_)
                  {
                    return;
                  }
                  void *local_gpu_buffer = nullptr;
                  cudaEvent_t ready_event = nullptr;
                  int pool_index = -1;
                  if (!nitros_sub_pool_->acquire(total_size, pool_index,
                                                 local_gpu_buffer, ready_event,
                                                 2))
                  {
                    return;
                  }

                  cudaError_t err = cudaMemcpyAsync(
                      local_gpu_buffer, gpu_data, total_size,
                      cudaMemcpyDeviceToDevice, cuda_sub_stream_);
                  if (err != cudaSuccess)
                  {
                    nitros_sub_pool_->release(pool_index);
                    return;
                  }
                  err = cudaEventRecord(ready_event, cuda_sub_stream_);
                  if (err != cudaSuccess)
                  {
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
                    while (!nv12_queue_.empty())
                    {
                      auto old = nv12_queue_.front();
                      nv12_queue_.pop();
                      if (nitros_sub_pool_)
                      {
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
                    while (draw_queue_.size() >= max_draw_queue_size_)
                    {
                      DrawItem old = std::move(draw_queue_.front());
                      draw_queue_.pop();
                      if (old.is_nv12 && nitros_sub_pool_)
                      {
                        nitros_sub_pool_->release(old.nv12_frame.pool_index);
                      }
                    }
                    draw_queue_.push(std::move(item));
                  }
                  draw_queue_cv_.notify_one();
                }),
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, qos_profile);
      }
      else
      {
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
                    [this](const sensor_msgs::msg::Image::SharedPtr msg)
                    {
                      if (!running_ || !msg)
                      {
                        return;
                      }

                      cv::Mat bgr;
                      const int width = static_cast<int>(msg->width);
                      const int height = static_cast<int>(msg->height);
                      if (width <= 0 || height <= 0)
                      {
                        return;
                      }

                      if (msg->encoding == "bgr8")
                      {
                        bgr = cv::Mat(height, width, CV_8UC3,
                                      const_cast<uint8_t *>(msg->data.data()),
                                      msg->step)
                                  .clone();
                      }
                      else if (msg->encoding == "rgb8")
                      {
                        cv::Mat rgb(height, width, CV_8UC3,
                                    const_cast<uint8_t *>(msg->data.data()),
                                    msg->step);
                        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                      }
                      else if (msg->encoding == "bgra8")
                      {
                        cv::Mat bgra(height, width, CV_8UC4,
                                     const_cast<uint8_t *>(msg->data.data()),
                                     msg->step);
                        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
                      }
                      else if (msg->encoding == "rgba8")
                      {
                        cv::Mat rgba(height, width, CV_8UC4,
                                     const_cast<uint8_t *>(msg->data.data()),
                                     msg->step);
                        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
                      }
                      else if (msg->encoding == "yuv422_yuy2" ||
                               msg->encoding == "yuyv" ||
                               msg->encoding == "yuy2")
                      {
                        cv::Mat yuy2(height, width, CV_8UC2,
                                     const_cast<uint8_t *>(msg->data.data()),
                                     msg->step);
                        cv::cvtColor(yuy2, bgr, cv::COLOR_YUV2BGR_YUY2);
                      }
                      else
                      {
                        return;
                      }

                      if (bgr.empty())
                      {
                        return;
                      }

                      {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        if (image_queue_.size() >= max_queue_size_)
                        {
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

      if (use_nitros_pub)
      {
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
      }
      else
      {
        const std::string camera_ns = GetSingleCameraNamespace(input_config_);
        image_raw_publisher_ =
            ros2_client_->getNode()->create_publisher<sensor_msgs::msg::Image>(
                "/" + camera_ns + "/image_perceptual/raw",
                rclcpp::QoS(1).reliable());
      }

      if (!input_config_.input_sizes.empty())
      {
        const auto sz = input_config_.input_sizes[0];
        if (sz.width > 0 && sz.height > 0)
        {
          const size_t nv12_size =
              static_cast<size_t>(sz.width) * sz.height * 2;
          nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
          nitros_publish_pool_->warmup(6);
        }
      }

      {
        auto node = ros2_client_->getNode();
        const int cam_id =
            (input_config_.camera_ids.empty() ||
             static_cast<size_t>(input_config_.idx) >=
                 input_config_.camera_ids.size())
                ? 0
                : input_config_
                      .camera_ids[static_cast<size_t>(input_config_.idx)];
        detection_distance_pub_ = node->create_publisher<std_msgs::msg::String>(
            "/camera_" + std::to_string(cam_id) + "/detection_distance",
            rclcpp::QoS(1).reliable());

        if (use_nitros_sub_)
        {
          bucket_bbox3d_marker_pub_ = node->create_publisher<
              visualization_msgs::msg::MarkerArray>(
              "/camera_" + std::to_string(cam_id) + "/bucket_bbox3d_markers",
              rclcpp::QoS(1).reliable());
        }

        const auto declare_string_param = [&](const std::string &name,
                                              const std::string &def)
        {
          try
          {
            if (!node->has_parameter(name))
            {
              node->declare_parameter<std::string>(name, def);
            }
          }
          catch (...)
          {
          }
          std::string out = def;
          try
          {
            node->get_parameter(name, out);
          }
          catch (...)
          {
          }
          return out;
        };

        const auto declare_int_param = [&](const std::string &name, int def)
        {
          try
          {
            if (!node->has_parameter(name))
            {
              node->declare_parameter<int>(name, def);
            }
          }
          catch (...)
          {
          }
          int out = def;
          try
          {
            node->get_parameter(name, out);
          }
          catch (...)
          {
          }
          return out;
        };

        const auto declare_double_param = [&](const std::string &name,
                                              double def)
        {
          try
          {
            if (!node->has_parameter(name))
            {
              node->declare_parameter<double>(name, def);
            }
          }
          catch (...)
          {
          }
          double out = def;
          try
          {
            node->get_parameter(name, out);
          }
          catch (...)
          {
          }
          return out;
        };

        const auto declare_int_param_alias = [&](const std::string &name,
                                                 const std::string &legacy_name,
                                                 int def)
        {
          int out = declare_int_param(name, def);
          if (legacy_name.empty())
          {
            return out;
          }
          try
          {
            if (!node->has_parameter(legacy_name))
            {
              node->declare_parameter<int>(legacy_name, def);
            }
          }
          catch (...)
          {
          }
          if (out != def)
          {
            return out;
          }
          try
          {
            node->get_parameter(legacy_name, out);
          }
          catch (...)
          {
          }
          return out;
        };

        const auto declare_double_param_alias = [&](
                                                    const std::string &name,
                                                    const std::string &legacy_name,
                                                    double def)
        {
          double out = declare_double_param(name, def);
          if (legacy_name.empty())
          {
            return out;
          }
          try
          {
            if (!node->has_parameter(legacy_name))
            {
              node->declare_parameter<double>(legacy_name, def);
            }
          }
          catch (...)
          {
          }
          if (out != def)
          {
            return out;
          }
          try
          {
            node->get_parameter(legacy_name, out);
          }
          catch (...)
          {
          }
          return out;
        };

        const std::string calib_result_path = declare_string_param(
            "calib_result_path",
            "/home/nvidia/ISAAC/src/isaac_ros_object_detection/config/calib_result.yaml");

        {
          std::ifstream ifs(calib_result_path);
          if (ifs.good())
          {
            std::string line;
            std::string model_str;
            double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
            double fx_ph = 0.0, fy_ph = 0.0, cx_ph = 0.0, cy_ph = 0.0;
            double fx_fe = 0.0, fy_fe = 0.0, cx_fe = 0.0, cy_fe = 0.0;
            std::array<double, 8> dist_ph{};
            std::array<double, 4> dist_fe{};
            bool have_ph_intr = false;
            bool have_fe_intr = false;
            std::array<double, 9> R{};
            std::array<double, 3> t{};
            bool have_fx = false, have_fy = false, have_cx = false,
                 have_cy = false;
            bool have_R = false, have_t = false;

            auto trim = [](std::string s)
            {
              const auto is_space = [](unsigned char c)
              {
                return std::isspace(c);
              };
              while (!s.empty() &&
                     is_space(static_cast<unsigned char>(s.front())))
              {
                s.erase(s.begin());
              }
              while (!s.empty() &&
                     is_space(static_cast<unsigned char>(s.back())))
              {
                s.pop_back();
              }
              return s;
            };

            auto parse_key_double = [&](const std::string &prefix, double &out,
                                        bool &have)
            {
              if (line.rfind(prefix, 0) != 0)
              {
                return;
              }
              const auto pos = line.find(':');
              if (pos == std::string::npos)
              {
                return;
              }
              try
              {
                out = std::stod(trim(line.substr(pos + 1)));
                have = true;
              }
              catch (...)
              {
              }
            };

            std::vector<double> rvals;
            std::vector<double> tvals;
            auto extract_doubles = [](const std::string &s,
                                      std::vector<double> &vals)
            {
              const char *p = s.c_str();
              while (*p)
              {
                char *end = nullptr;
                const double v = std::strtod(p, &end);
                if (end != nullptr && end != p)
                {
                  vals.push_back(v);
                  p = end;
                }
                else
                {
                  ++p;
                }
              }
            };

            bool reading_R = false;
            bool reading_t = false;
            std::string section;
            while (std::getline(ifs, line))
            {
              line = trim(line);
              if (line.empty() || line[0] == '#')
              {
                continue;
              }
              const auto colon = line.find(':');
              if (colon != std::string::npos)
              {
                const std::string key = trim(line.substr(0, colon));
                const std::string val = trim(line.substr(colon + 1));
                if (!key.empty() && val.empty())
                {
                  std::string s = key;
                  for (auto &c : s)
                  {
                    c = static_cast<char>(
                        std::tolower(static_cast<unsigned char>(c)));
                  }
                  section = s;
                }
                if (key == "model" || key == "cam_model")
                {
                  if (!val.empty())
                  {
                    model_str = val;
                  }
                }
              }

              parse_key_double("cam_fx", fx, have_fx);
              parse_key_double("cam_fy", fy, have_fy);
              parse_key_double("cam_cx", cx, have_cx);
              parse_key_double("cam_cy", cy, have_cy);
              parse_key_double("fx", fx, have_fx);
              parse_key_double("fy", fy, have_fy);
              parse_key_double("cx", cx, have_cx);
              parse_key_double("cy", cy, have_cy);

              if (section == "pinhole")
              {
                bool tmp = false;
                parse_key_double("fx", fx_ph, tmp);
                if (tmp)
                {
                  have_ph_intr = true;
                }
                parse_key_double("fy", fy_ph, tmp);
                if (tmp)
                {
                  have_ph_intr = true;
                }
                parse_key_double("cx", cx_ph, tmp);
                if (tmp)
                {
                  have_ph_intr = true;
                }
                parse_key_double("cy", cy_ph, tmp);
                if (tmp)
                {
                  have_ph_intr = true;
                }
                bool h = false;
                parse_key_double("k1", dist_ph[0], h);
                parse_key_double("k2", dist_ph[1], h);
                parse_key_double("p1", dist_ph[2], h);
                parse_key_double("p2", dist_ph[3], h);
                parse_key_double("k3", dist_ph[4], h);
                parse_key_double("k4", dist_ph[5], h);
                parse_key_double("k5", dist_ph[6], h);
                parse_key_double("k6", dist_ph[7], h);
              }
              else if (section == "fisheye" || section == "fish_eye" ||
                       section == "fish-eye")
              {
                bool tmp = false;
                parse_key_double("fx", fx_fe, tmp);
                if (tmp)
                {
                  have_fe_intr = true;
                }
                parse_key_double("fy", fy_fe, tmp);
                if (tmp)
                {
                  have_fe_intr = true;
                }
                parse_key_double("cx", cx_fe, tmp);
                if (tmp)
                {
                  have_fe_intr = true;
                }
                parse_key_double("cy", cy_fe, tmp);
                if (tmp)
                {
                  have_fe_intr = true;
                }
                bool h = false;
                parse_key_double("k1", dist_fe[0], h);
                parse_key_double("k2", dist_fe[1], h);
                parse_key_double("k3", dist_fe[2], h);
                parse_key_double("k4", dist_fe[3], h);
              }
              else
              {
                bool h = false;
                parse_key_double("k1", dist_ph[0], h);
                parse_key_double("k2", dist_ph[1], h);
                parse_key_double("p1", dist_ph[2], h);
                parse_key_double("p2", dist_ph[3], h);
                parse_key_double("k3", dist_ph[4], h);
                parse_key_double("k4", dist_ph[5], h);
                parse_key_double("k5", dist_ph[6], h);
                parse_key_double("k6", dist_ph[7], h);
                parse_key_double("cam_d0", dist_ph[0], h);
                parse_key_double("cam_d1", dist_ph[1], h);
                parse_key_double("cam_d2", dist_ph[2], h);
                parse_key_double("cam_d3", dist_ph[3], h);
              }

              if ((!have_fx || !have_fy || !have_cx || !have_cy) &&
                  (line.rfind("K:", 0) == 0 || line.rfind("cam_K", 0) == 0 ||
                   line.rfind("K =", 0) == 0 || line.rfind("K=", 0) == 0))
              {
                std::vector<double> kvals;
                extract_doubles(line, kvals);
                if (kvals.size() >= 6)
                {
                  fx = kvals[0];
                  cx = kvals[2];
                  fy = kvals[4];
                  cy = kvals[5];
                  have_fx = have_fy = have_cx = have_cy = true;
                }
              }

              const bool is_R_line =
                  (line.rfind("Rcl", 0) == 0) || (line.rfind("R_cl", 0) == 0) ||
                  (line.rfind("R:", 0) == 0) || (line.rfind("R =", 0) == 0) ||
                  (line.rfind("R=", 0) == 0);
              const bool is_t_line =
                  (line.rfind("Pcl", 0) == 0) || (line.rfind("tcl", 0) == 0) ||
                  (line.rfind("t:", 0) == 0) || (line.rfind("t =", 0) == 0) ||
                  (line.rfind("t=", 0) == 0) || (line.rfind("P:", 0) == 0) ||
                  (line.rfind("P =", 0) == 0) || (line.rfind("P=", 0) == 0);

              if (is_R_line)
              {
                reading_R = true;
                rvals.clear();
              }
              if (is_t_line)
              {
                reading_t = true;
                tvals.clear();
              }

              if (reading_R && !have_R)
              {
                extract_doubles(line, rvals);
                if (rvals.size() >= 9)
                {
                  for (size_t i = 0; i < 9; ++i)
                  {
                    R[i] = rvals[i];
                  }
                  have_R = true;
                  reading_R = false;
                }
              }

              if (reading_t && !have_t)
              {
                extract_doubles(line, tvals);
                if (tvals.size() >= 3)
                {
                  for (size_t i = 0; i < 3; ++i)
                  {
                    t[i] = tvals[i];
                  }
                  have_t = true;
                  reading_t = false;
                }
              }
            }

            auto normalize_model = [](std::string s)
            {
              auto trim_local = [](std::string x)
              {
                const auto is_space = [](unsigned char c)
                {
                  return std::isspace(c);
                };
                while (!x.empty() &&
                       is_space(static_cast<unsigned char>(x.front())))
                {
                  x.erase(x.begin());
                }
                while (!x.empty() &&
                       is_space(static_cast<unsigned char>(x.back())))
                {
                  x.pop_back();
                }
                return x;
              };
              s = trim_local(s);
              if (!s.empty() && (s.front() == '"' || s.front() == '\''))
              {
                s.erase(s.begin());
              }
              if (!s.empty() && (s.back() == '"' || s.back() == '\''))
              {
                s.pop_back();
              }
              for (auto &c : s)
              {
                c = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
              }
              return s;
            };

            const std::string m = normalize_model(model_str);
            CamLidarCalib::CameraModel model =
                CamLidarCalib::CameraModel::PinholeRational;
            if (m.find("fish") != std::string::npos)
            {
              model = CamLidarCalib::CameraModel::FishEye;
            }

            if (model == CamLidarCalib::CameraModel::FishEye && have_fe_intr)
            {
              fx = fx_fe;
              fy = fy_fe;
              cx = cx_fe;
              cy = cy_fe;
              have_fx = have_fy = have_cx = have_cy = true;
            }
            else if (have_ph_intr)
            {
              fx = fx_ph;
              fy = fy_ph;
              cx = cx_ph;
              cy = cy_ph;
              have_fx = have_fy = have_cx = have_cy = true;
            }

            cam_lidar_calib_.valid = (have_R && have_t) && (have_fx && have_fy);
            cam_lidar_calib_.model = model;
            cam_lidar_calib_.fx = fx;
            cam_lidar_calib_.fy = fy;
            cam_lidar_calib_.cx = cx;
            cam_lidar_calib_.cy = cy;
            cam_lidar_calib_.dist_pinhole = dist_ph;
            cam_lidar_calib_.dist_fisheye = dist_fe;
            cam_lidar_calib_.R = R;
            cam_lidar_calib_.t = t;
          }
          else
          {
            cam_lidar_calib_.valid = false;
          }
        }

        const std::string livox_topic =
            declare_string_param("livox_topic", "/livox/lidar");
        const int livox_xfer_format = declare_int_param("livox_xfer_format", 1);

        distance_cfg_.tooth_band_height_ratio =
            declare_double_param("distance_tooth_band_height_ratio", 0.12);
        distance_cfg_.tooth_band_x_margin_ratio =
            declare_double_param("distance_tooth_band_x_margin_ratio", 0.08);
        distance_cfg_.tooth_band_y_offset_ratio =
            declare_double_param("distance_tooth_band_y_offset_ratio", 0.0);
        distance_cfg_.near_band_m =
            declare_double_param("distance_near_band_m", 0.08);
        distance_cfg_.hold_max_dt_ms =
            declare_double_param("distance_hold_max_dt_ms", 250.0);
        distance_cfg_.outlier_mad_k =
            declare_double_param("distance_outlier_mad_k", 3.5);
        distance_cfg_.outlier_min_abs_m =
            declare_double_param("distance_outlier_min_abs_m", 0.02);
        distance_cfg_.min_z = declare_double_param("distance_min_z", 0.1);
        distance_cfg_.max_z = declare_double_param("distance_max_z", 30.0);
        distance_cfg_.max_age_ms =
            declare_double_param("distance_max_age_ms", 200.0);
        distance_cfg_.min_points = declare_int_param("distance_min_points", 5);

        bbox3d_cfg_.enable =
            (declare_int_param_alias("bbox3d_enable", "bucket_bbox3d_enable", 0) !=
             0); // 是否启用3D框估计
        bbox3d_cfg_.window_ms = declare_double_param_alias(
            "bbox3d_window_ms", "bucket_bbox3d_window_ms", 1000.0);
        bbox3d_cfg_.max_frames =
            declare_int_param_alias("bbox3d_max_frames",
                                    "bucket_bbox3d_max_frames", 20);
        bbox3d_cfg_.min_points =
            declare_int_param_alias("bbox3d_min_points",
                                    "bucket_bbox3d_min_points", 80);
        bbox3d_cfg_.max_points =
            declare_int_param_alias("bbox3d_max_points",
                                    "bucket_bbox3d_max_points", 200000);
        bbox3d_cfg_.max_age_ms = declare_double_param_alias(
            "bbox3d_max_age_ms", "bucket_bbox3d_max_age_ms", 300.0);
        bbox3d_cfg_.depth_gate_m = declare_double_param_alias(
            "bbox3d_depth_gate_m", "bucket_bbox3d_depth_gate_m", 1.2);
        bbox3d_cfg_.depth_bin_m = declare_double_param_alias(
            "bbox3d_depth_bin_m", "bucket_bbox3d_depth_bin_m", 0.2);
        bbox3d_cfg_.roi_shrink_ratio = declare_double_param_alias(
            "bbox3d_roi_shrink_ratio", "bucket_bbox3d_roi_shrink_ratio", 0.08);
        bbox3d_cfg_.hold_ms = declare_double_param_alias(
            "bbox3d_hold_ms", "bucket_bbox3d_hold_ms", 500.0);

        rclcpp::QoS pc_qos(1);
        pc_qos.best_effort();
        pc_qos.keep_last(1);

        rclcpp::SubscriptionOptions pc_sub_options;
        pc_sub_options.callback_group = callback_group_;
        if (livox_xfer_format == 0)
        {
          livox_pc2_sub_ =
              node->create_subscription<sensor_msgs::msg::PointCloud2>(
                  livox_topic, pc_qos,
                  [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
                  {
                    this->handlePointCloud2(msg);
                  },
                  pc_sub_options);
        }
        else
        {
          livox_custom_sub_ =
              node->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                  livox_topic, pc_qos,
                  [this](
                      const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
                  {
                    this->handleLivoxCustomMsg(msg);
                  },
                  pc_sub_options);
        }
      }
    }
    else
    {
      if (!openInputStream(image_processing_config_, input_config_))
      {
        sample::gLogError << "Failed to open input stream." << std::endl;
        return;
      }
    }
    general_config_.is_init = true;
  }
}

void CameraCombinedWithLidarPipelineManager::cleanupResources()
{
  // 智能指针显式资源释放
  yolo_.reset();
  if (model_config_.need_second_detection_)
  {
    second_yolo_.reset();
    // 释放二次目标检测线程池资源
    model_config_.bucket_thread_pool_.reset();
  }
  if (model_config_.need_classifier_)
  {
    // 释放分类器线程池资源
    model_config_.classifier_thread_pool_.reset();
  }
  trt_classifier_.reset();
  undistorter_.reset();
  mqtt_publisher_.reset();

  nitros_image_subscriber_.reset();
  nitros_image_publisher_.reset();
  image_raw_publisher_.reset();
  livox_pc2_sub_.reset();
  livox_custom_sub_.reset();
  detection_distance_pub_.reset();
  bucket_bbox3d_marker_pub_.reset();
  nitros_publish_pool_.reset();
  {
    std::lock_guard<std::mutex> lock(nv12_queue_mutex_);
    while (!nv12_queue_.empty())
    {
      auto f = nv12_queue_.front();
      nv12_queue_.pop();
      if (nitros_sub_pool_)
      {
        nitros_sub_pool_->release(f.pool_index);
      }
    }
  }
  nitros_sub_pool_.reset();
  callback_group_.reset();
  {
    std::lock_guard<std::mutex> lk(lidar_cloud_mutex_);
    lidar_cloud_cache_ = LidarCloudCache{};
  }
  {
    std::lock_guard<std::mutex> lk(lidar_cloud_history_mutex_);
    lidar_cloud_history_.clear();
    last_lidar_frame_id_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(bucket_bbox3d_cache_mutex_);
    bucket_bbox3d_cache_.clear();
    bucket_bbox3d_cache_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  }
  {
    std::lock_guard<std::mutex> lk(distance_req_mutex_);
    distance_req_ = DistanceRequest{};
  }
  {
    std::lock_guard<std::mutex> lk(distance_cache_mutex_);
    distance_cache_ = DistanceBundle{};
    distance_cache_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  }

  if (cuda_stream_)
  {
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }
  if (cuda_sub_stream_)
  {
    cudaStreamDestroy(cuda_sub_stream_);
    cuda_sub_stream_ = nullptr;
  }
  if (cuda_publish_stream_)
  {
    cudaStreamDestroy(cuda_publish_stream_);
    cuda_publish_stream_ = nullptr;
  }
  if (cuda_nv12_sub_buffer_)
  {
    cudaFree(cuda_nv12_sub_buffer_);
    cuda_nv12_sub_buffer_ = nullptr;
    cuda_nv12_sub_buffer_size_ = 0;
  }
  if (cuda_bgr_sub_buffer_)
  {
    cudaFree(cuda_bgr_sub_buffer_);
    cuda_bgr_sub_buffer_ = nullptr;
    cuda_bgr_sub_buffer_size_ = 0;
  }
  if (cuda_bgr_publish_buffer_)
  {
    cudaFree(cuda_bgr_publish_buffer_);
    cuda_bgr_publish_buffer_ = nullptr;
    cuda_bgr_publish_buffer_size_ = 0;
  }
  if (cuda_bbox_buffer_)
  {
    cudaFree(cuda_bbox_buffer_);
    cuda_bbox_buffer_ = nullptr;
    cuda_bbox_buffer_size_ = 0;
  }

  // 释放视频捕获资源
  if (image_processing_config_.captures[input_config_.idx].isOpened())
  {
    image_processing_config_.captures[input_config_.idx].release();
  }

  // 清理ROS2资源
  ros2_client_.reset();
  if (input_config_.ros2_enabled && rclcpp::ok() && !parent_node_)
  {
    rclcpp::shutdown();
  }

  sample::gLogInfo << "All threads have finished processing." << std::endl;
}

bool CameraCombinedWithLidarPipelineManager::publishNitrosImage(
    const cv::Mat &bgr_frame)
{
  if (!input_config_.ros2_enabled || !ros2_client_ ||
      !nitros_image_publisher_ || bgr_frame.empty())
  {
    return false;
  }
  if (cuda_stream_ == nullptr)
  {
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
                        "publish bgr"))
  {
    return false;
  }

  const size_t nv12_size = static_cast<size_t>(width) * height * 2;
  if (!nitros_publish_pool_ || nitros_publish_pool_->size_bytes != nv12_size)
  {
    nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
    nitros_publish_pool_->warmup(2);
  }

  void *publish_buffer = nitros_publish_pool_->acquire();
  if (!publish_buffer)
  {
    return false;
  }

  cudaError_t cuda_err =
      cudaMemcpyAsync(cuda_bgr_publish_buffer_, bgr_frame.data, bgr_size,
                      cudaMemcpyHostToDevice, cuda_stream_);
  if (cuda_err != cudaSuccess)
  {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }

  cuda_err = ConvertBGRToGxfNV12(publish_buffer, cuda_bgr_publish_buffer_,
                                 width, height, y_stride, uv_stride, bgr_stride,
                                 cuda_stream_);
  if (cuda_err != cudaSuccess)
  {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }

  cuda_err = cudaStreamSynchronize(cuda_stream_);
  if (cuda_err != cudaSuccess)
  {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }

  std_msgs::msg::Header header;
  header.stamp = ros2_client_->getNode()->now();
  const int cam_id =
      input_config_.camera_ids.empty() ? 0 : input_config_.camera_ids[0];
  header.frame_id = "camera_" + std::to_string(cam_id);

  try
  {
    auto nitros_image = BuildNv12NitrosImageWithPool(
        header, static_cast<uint32_t>(height), static_cast<uint32_t>(width),
        publish_buffer, y_stride, uv_stride, nitros_publish_pool_);
    nitros_image_publisher_->publish(nitros_image);
  }
  catch (...)
  {
    nitros_publish_pool_->release(publish_buffer);
    return false;
  }
  return true;
}

bool CameraCombinedWithLidarPipelineManager::publishNitrosNv12WithOverlay(
    const Nv12GpuFrame &frame, const std::vector<std::vector<Box>> &bboxes)
{
  if (!input_config_.ros2_enabled || !ros2_client_ ||
      !nitros_image_publisher_ || !frame.gpu_buffer || frame.width <= 0 ||
      frame.height <= 0 || frame.y_stride <= 0 || frame.uv_stride <= 0 ||
      frame.size_bytes == 0)
  {
    return false;
  }

  auto logger = ros2_client_->getNode()->get_logger();
  if (cuda_publish_stream_ == nullptr)
  {
    cudaStreamCreateWithFlags(&cuda_publish_stream_, cudaStreamNonBlocking);
  }

  const size_t nv12_size = frame.size_bytes;
  if (!nitros_publish_pool_ || nitros_publish_pool_->size_bytes != nv12_size)
  {
    nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
    nitros_publish_pool_->warmup(6);
  }
  auto pool = nitros_publish_pool_;

  void *publish_buffer = pool ? pool->acquire() : nullptr;
  if (!publish_buffer)
  {
    return false;
  }

  if (frame.ready_event)
  {
    cudaStreamWaitEvent(cuda_publish_stream_, frame.ready_event, 0);
  }
  cudaError_t perr =
      cudaMemcpyAsync(publish_buffer, frame.gpu_buffer, nv12_size,
                      cudaMemcpyDeviceToDevice, cuda_publish_stream_);

  std::vector<GxfBBox> people_bbs;
  std::vector<GxfCircleMark> marks;
  static thread_local std::vector<GxfDashedLine> dashed_lines;
  static thread_local std::vector<GxfTextMark> text_marks;
  dashed_lines.clear();
  text_marks.clear();

  if (perr == cudaSuccess && !bboxes.empty() && !bboxes[0].empty())
  {
    const auto &boxes0 = bboxes[0];
    people_bbs.reserve(boxes0.size());
    marks.reserve(boxes0.size());
    dashed_lines.reserve(boxes0.size() * 2);
    text_marks.reserve(boxes0.size() * 2);

    const rclcpp::Time stamp(frame.stamp_sec, frame.stamp_nanosec,
                             RCL_ROS_TIME);
    submitDistanceRequest(stamp, boxes0);
    DistanceBundle bundle;
    if (!tryGetDistanceCache(stamp, bundle))
    {
      bundle.objects.clear();
    }

    for (const auto &b : boxes0)
    {
      if (b.label < 0)
      {
        continue;
      }

      if (b.label == 0 || b.label == 11)
      {
        people_bbs.push_back({static_cast<int>(std::lround(b.left)),
                              static_cast<int>(std::lround(b.top)),
                              static_cast<int>(std::lround(b.right)),
                              static_cast<int>(std::lround(b.bottom))});
        continue;
      }

      uint8_t yv = 0;
      uint8_t uv = 0;
      uint8_t vv = 0;
      if (b.label == 2)
      {
        yv = 145;
        uv = 54;
        vv = 34;
      }
      else if (b.label == 3)
      {
        yv = 100;
        uv = 90;
        vv = 240;
      }
      else
      {
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

    const int line_len_px = 120;
    const int dash_len = 8;
    const int gap_len = 6;
    const int thickness = 1;
    const uint8_t hy = 210;
    const uint8_t hu = 16;
    const uint8_t hv = 146;
    const uint8_t vy = 180;
    const uint8_t vu = 212;
    const uint8_t vv2 = 234;

    auto fill_text = [](GxfTextMark &tm, const char *s)
    {
      std::memset(tm.text, 0, sizeof(tm.text));
      if (!s)
      {
        return;
      }
      size_t n = 0;
      while (n + 1 < sizeof(tm.text) && s[n] != '\0')
      {
        n++;
      }
      if (n > 0)
      {
        std::memcpy(tm.text, s, n);
      }
      tm.text[n] = '\0';
    };

    for (const auto &m : bundle.objects)
    {
      if (!m.has_distance)
      {
        continue;
      }
      const int cx_i =
          static_cast<int>(std::lround(static_cast<double>(m.center_u)));
      const int cy_i =
          static_cast<int>(std::lround(static_cast<double>(m.center_v)));
      if (cx_i < 0 || cx_i >= frame.width || cy_i < 0 || cy_i >= frame.height)
      {
        continue;
      }

      const int x1 = std::min(cx_i + line_len_px, frame.width - 1);
      const int y1 = std::min(cy_i + line_len_px, frame.height - 1);

      dashed_lines.push_back(
          {cx_i, cy_i, x1, cy_i, dash_len, gap_len, thickness, hy, hu, hv});
      dashed_lines.push_back(
          {cx_i, cy_i, cx_i, y1, dash_len, gap_len, thickness, vy, vu, vv2});

      char d_text[16];
      char h_text[16];
      std::snprintf(d_text, sizeof(d_text), "D=%.2fm",
                    static_cast<double>(m.depth_m));
      std::snprintf(h_text, sizeof(h_text), "H=%.2fm",
                    static_cast<double>(m.height_m));

      const int dx = std::min(x1 + 6, frame.width - 1);
      const int dy = std::max(cy_i - 10, 0);
      const int hx = std::min(cx_i + 6, frame.width - 1);
      const int hy0 = std::max(y1 - 12, 0);

      GxfTextMark dm{};
      dm.x = dx;
      dm.y = dy;
      dm.scale = 2;
      dm.y_val = hy;
      dm.u_val = hu;
      dm.v_val = hv;
      fill_text(dm, d_text);
      text_marks.push_back(dm);

      GxfTextMark hm{};
      hm.x = hx;
      hm.y = hy0;
      hm.scale = 2;
      hm.y_val = vy;
      hm.u_val = vu;
      hm.v_val = vv2;
      fill_text(hm, h_text);
      text_marks.push_back(hm);
    }

    // 3D envelope wireframe (projected to image).
    for (const auto &bbox3d : bundle.bbox3d_objects)
    {
      if (!bbox3d.valid)
      {
        continue;
      }
      const uint8_t by = 145;
      const uint8_t bu = 54;
      const uint8_t bv = 34;
      const int dash_len_bbox = 1024;
      const int gap_len_bbox = 0;
      const int thickness_bbox = 2;

      auto add_edge = [&](int i, int j)
      {
        if (i < 0 || j < 0 || i >= 8 || j >= 8)
        {
          return;
        }
        if (!bbox3d.corners_uv_valid[static_cast<size_t>(i)] ||
            !bbox3d.corners_uv_valid[static_cast<size_t>(j)])
        {
          return;
        }
        const auto &p0 = bbox3d.corners_uv[static_cast<size_t>(i)];
        const auto &p1 = bbox3d.corners_uv[static_cast<size_t>(j)];
        if (!(std::isfinite(p0.x) && std::isfinite(p0.y) && std::isfinite(p1.x) &&
              std::isfinite(p1.y)))
        {
          return;
        }
        const int x0i = std::max(
            0, std::min(static_cast<int>(std::lround(p0.x)), frame.width - 1));
        const int y0i = std::max(
            0, std::min(static_cast<int>(std::lround(p0.y)), frame.height - 1));
        const int x1i = std::max(
            0, std::min(static_cast<int>(std::lround(p1.x)), frame.width - 1));
        const int y1i = std::max(
            0, std::min(static_cast<int>(std::lround(p1.y)), frame.height - 1));
        dashed_lines.push_back({x0i, y0i, x1i, y1i, dash_len_bbox, gap_len_bbox,
                                thickness_bbox, by, bu, bv});
      };

      add_edge(0, 1);
      add_edge(1, 2);
      add_edge(2, 3);
      add_edge(3, 0);
      add_edge(4, 5);
      add_edge(5, 6);
      add_edge(6, 7);
      add_edge(7, 4);
      add_edge(0, 4);
      add_edge(1, 5);
      add_edge(2, 6);
      add_edge(3, 7);
    }
  }

  const size_t need_bboxes = people_bbs.size() * sizeof(GxfBBox);
  const size_t need_marks = marks.size() * sizeof(GxfCircleMark);
  const size_t need_lines = dashed_lines.size() * sizeof(GxfDashedLine);
  const size_t need_texts = text_marks.size() * sizeof(GxfTextMark);
  const size_t need_max = std::max(std::max(need_bboxes, need_marks),
                                   std::max(need_lines, need_texts));
  if (perr == cudaSuccess && need_max > 0)
  {
    if (!EnsureCudaBuffer(&cuda_bbox_buffer_, cuda_bbox_buffer_size_, need_max,
                          logger, "nv12 draw buffer"))
    {
      perr = cudaErrorMemoryAllocation;
    }
  }

  if (perr == cudaSuccess && !people_bbs.empty())
  {
    perr = cudaMemcpyAsync(cuda_bbox_buffer_, people_bbs.data(), need_bboxes,
                           cudaMemcpyHostToDevice, cuda_publish_stream_);
    if (perr == cudaSuccess)
    {
      constexpr uint8_t y_val = 100;
      constexpr uint8_t u_val = 90;
      constexpr uint8_t v_val = 240;
      perr = DrawGxfNV12Bboxes(publish_buffer, frame.width, frame.height,
                               frame.y_stride, frame.uv_stride,
                               static_cast<const GxfBBox *>(cuda_bbox_buffer_),
                               static_cast<int>(people_bbs.size()), 2, y_val,
                               u_val, v_val, cuda_publish_stream_);
    }
  }

  if (perr == cudaSuccess && !marks.empty())
  {
    perr = cudaMemcpyAsync(cuda_bbox_buffer_, marks.data(), need_marks,
                           cudaMemcpyHostToDevice, cuda_publish_stream_);
    if (perr == cudaSuccess)
    {
      perr = DrawGxfNV12CircleMarks(
          publish_buffer, frame.width, frame.height, frame.y_stride,
          frame.uv_stride,
          static_cast<const GxfCircleMark *>(cuda_bbox_buffer_),
          static_cast<int>(marks.size()), cuda_publish_stream_);
    }
  }

  if (perr == cudaSuccess && !dashed_lines.empty())
  {
    perr = cudaMemcpyAsync(cuda_bbox_buffer_, dashed_lines.data(), need_lines,
                           cudaMemcpyHostToDevice, cuda_publish_stream_);
    if (perr == cudaSuccess)
    {
      perr = DrawGxfNV12DashedLines(
          publish_buffer, frame.width, frame.height, frame.y_stride,
          frame.uv_stride,
          static_cast<const GxfDashedLine *>(cuda_bbox_buffer_),
          static_cast<int>(dashed_lines.size()), cuda_publish_stream_);
    }
  }

  if (perr == cudaSuccess && !text_marks.empty())
  {
    perr = cudaMemcpyAsync(cuda_bbox_buffer_, text_marks.data(), need_texts,
                           cudaMemcpyHostToDevice, cuda_publish_stream_);
    if (perr == cudaSuccess)
    {
      perr = DrawGxfNV12TextMarks(
          publish_buffer, frame.width, frame.height, frame.y_stride,
          frame.uv_stride, static_cast<const GxfTextMark *>(cuda_bbox_buffer_),
          static_cast<int>(text_marks.size()), cuda_publish_stream_);
    }
  }

  const int fps_x10 = static_cast<int>(std::lround(fps_value_ * 10.0));
  if (perr == cudaSuccess)
  {
    perr = DrawGxfNV12Fps(publish_buffer, frame.width, frame.height,
                          frame.y_stride, frame.uv_stride, fps_x10,
                          cuda_publish_stream_);
  }

  if (perr == cudaSuccess)
  {
    perr = cudaStreamSynchronize(cuda_publish_stream_);
  }
  if (perr != cudaSuccess)
  {
    if (pool)
    {
      pool->release(publish_buffer);
    }
    else
    {
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

  try
  {
    auto nitros_image = BuildNv12NitrosImageWithPool(
        header, static_cast<uint32_t>(frame.height),
        static_cast<uint32_t>(frame.width), publish_buffer, frame.y_stride,
        frame.uv_stride, pool);
    nitros_image_publisher_->publish(nitros_image);
  }
  catch (...)
  {
    if (pool)
    {
      pool->release(publish_buffer);
    }
    else
    {
      cudaFree(publish_buffer);
    }
    return false;
  }
  return true;
}

void CameraCombinedWithLidarPipelineManager::drawThread()
{
  while (running_)
  {
    DrawItem item;
    {
      std::unique_lock<std::mutex> lock(draw_queue_mutex_);
      draw_queue_cv_.wait(lock,
                          [this]
                          { return !draw_queue_.empty() || !running_; });
      if (!running_ && draw_queue_.empty())
      {
        return;
      }
      while (draw_queue_.size() > 1)
      {
        DrawItem old = std::move(draw_queue_.front());
        draw_queue_.pop();
        if (old.is_nv12 && nitros_sub_pool_)
        {
          nitros_sub_pool_->release(old.nv12_frame.pool_index);
        }
      }
      item = std::move(draw_queue_.front());
      draw_queue_.pop();
    }

    if (item.is_nv12)
    {
      if (input_config_.ros2_enabled && ros2_client_)
      {
        const auto now_tp = std::chrono::steady_clock::now();
        fps_window_frames_ += 1;
        const double sec =
            std::chrono::duration<double>(now_tp - fps_window_start_tp_)
                .count();
        if (sec >= 1.0)
        {
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
            EncodingFormat::NITROS_NV12)
        {
          publishNitrosNv12WithOverlay(item.nv12_frame, overlay_boxes);
        }

        if (image_raw_publisher_ || general_config_.show_gui)
        {
          if (cuda_stream_ == nullptr)
          {
            cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
          }

          cv::Mat bgr(item.nv12_frame.height, item.nv12_frame.width, CV_8UC3);
          const int bgr_stride = item.nv12_frame.width * 3;
          const size_t bgr_size =
              static_cast<size_t>(bgr_stride) * item.nv12_frame.height;
          auto logger = ros2_client_->getNode()->get_logger();
          if (EnsureCudaBuffer(&cuda_bgr_sub_buffer_, cuda_bgr_sub_buffer_size_,
                               bgr_size, logger, "bgr draw"))
          {
            cudaStreamWaitEvent(cuda_stream_, item.nv12_frame.ready_event);
            cudaError_t err = ConvertGxfNV12ToBGR(
                cuda_bgr_sub_buffer_, item.nv12_frame.gpu_buffer,
                item.nv12_frame.width, item.nv12_frame.height,
                item.nv12_frame.y_stride, item.nv12_frame.uv_stride, bgr_stride,
                cuda_stream_);
            if (err == cudaSuccess)
            {
              err = cudaMemcpyAsync(bgr.data, cuda_bgr_sub_buffer_, bgr_size,
                                    cudaMemcpyDeviceToHost, cuda_stream_);
              if (err == cudaSuccess)
              {
                cudaStreamSynchronize(cuda_stream_);
              }
            }
          }

          const rclcpp::Time stamp(item.nv12_frame.stamp_sec,
                                   item.nv12_frame.stamp_nanosec, RCL_ROS_TIME);
          if (!overlay_boxes.empty() && !overlay_boxes[0].empty() &&
              !bgr.empty())
          {
            submitDistanceRequest(stamp, overlay_boxes[0]);
            DistanceBundle bundle;
            if (!tryGetDistanceCache(stamp, bundle))
            {
              bundle.objects.clear();
            }

            const cv::Scalar box_color = BgrFromYuv(100, 90, 240);
            const cv::Scalar mark2_color = BgrFromYuv(145, 54, 34);
            const cv::Scalar mark3_color = BgrFromYuv(100, 90, 240);

            for (const auto &b : overlay_boxes[0])
            {
              if (b.label == 0 || b.label == 11)
              {
                const int x1 =
                    std::max(0, std::min(static_cast<int>(std::lround(b.left)),
                                         bgr.cols - 1));
                const int y1 =
                    std::max(0, std::min(static_cast<int>(std::lround(b.top)),
                                         bgr.rows - 1));
                const int x2 =
                    std::max(0, std::min(static_cast<int>(std::lround(b.right)),
                                         bgr.cols - 1));
                const int y2 = std::max(
                    0, std::min(static_cast<int>(std::lround(b.bottom)),
                                bgr.rows - 1));
                cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2),
                              box_color, 2, cv::LINE_AA);
              }
              else if (b.label == 2 || b.label == 3)
              {
                const int cx = static_cast<int>(
                    std::lround((static_cast<double>(b.left) +
                                 static_cast<double>(b.right)) *
                                0.5));
                const int cy = static_cast<int>(
                    std::lround((static_cast<double>(b.top) +
                                 static_cast<double>(b.bottom)) *
                                0.5));
                const int x = std::max(0, std::min(cx, bgr.cols - 1));
                const int y = std::max(0, std::min(cy, bgr.rows - 1));
                const cv::Scalar c = (b.label == 2) ? mark2_color : mark3_color;
                cv::circle(bgr, cv::Point(x, y), 5, c, -1, cv::LINE_AA);
              }
            }

            const int line_len_px = 120;
            const int dash_len = 8;
            const int gap_len = 6;
            const int thickness = 1;
            const cv::Scalar horiz_color = BgrFromYuv(210, 16, 146);
            const cv::Scalar vert_color = BgrFromYuv(180, 212, 234);

            for (const auto &m : bundle.objects)
            {
              if (!m.has_distance)
              {
                continue;
              }
              const int cx = static_cast<int>(
                  std::lround(static_cast<double>(m.center_u)));
              const int cy = static_cast<int>(
                  std::lround(static_cast<double>(m.center_v)));
              if (cx < 0 || cx >= bgr.cols || cy < 0 || cy >= bgr.rows)
              {
                continue;
              }

              const int x1 = std::min(cx + line_len_px, bgr.cols - 1);
              const int y1 = std::min(cy + line_len_px, bgr.rows - 1);

              DrawDashedHLine(bgr, cx, x1, cy, horiz_color, thickness, dash_len,
                              gap_len);
              DrawDashedVLine(bgr, cx, cy, y1, vert_color, thickness, dash_len,
                              gap_len);

              const std::string d_text = cv::format("D=%.2fm", m.depth_m);
              const std::string h_text = cv::format("H=%.2fm", m.height_m);

              const int dx = std::min(x1 + 6, bgr.cols - 1);
              const int dy = std::max(cy - 10, 0);
              const int hx = std::min(cx + 6, bgr.cols - 1);
              const int hy = std::max(y1 - 12, 0);

              cv::putText(bgr, d_text, cv::Point(dx, dy),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, horiz_color, 1,
                          cv::LINE_AA);
              cv::putText(bgr, h_text, cv::Point(hx, hy),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, vert_color, 1,
                          cv::LINE_AA);
            }

            for (const auto &bbox3d : bundle.bbox3d_objects)
            {
              if (!bbox3d.valid)
              {
                continue;
              }
              const cv::Scalar c3d = BgrFromYuv(80, 90, 240);
              auto draw_edge = [&](int i, int j)
              {
                if (i < 0 || j < 0 || i >= 8 || j >= 8)
                {
                  return;
                }
                if (!bbox3d.corners_uv_valid[static_cast<size_t>(i)] ||
                    !bbox3d.corners_uv_valid[static_cast<size_t>(j)])
                {
                  return;
                }
                const auto &p0 = bbox3d.corners_uv[static_cast<size_t>(i)];
                const auto &p1 = bbox3d.corners_uv[static_cast<size_t>(j)];
                const int x0i = std::max(
                    0, std::min(static_cast<int>(std::lround(p0.x)), bgr.cols - 1));
                const int y0i = std::max(
                    0, std::min(static_cast<int>(std::lround(p0.y)), bgr.rows - 1));
                const int x1i = std::max(
                    0, std::min(static_cast<int>(std::lround(p1.x)), bgr.cols - 1));
                const int y1i = std::max(
                    0, std::min(static_cast<int>(std::lround(p1.y)), bgr.rows - 1));
                cv::line(bgr, cv::Point(x0i, y0i), cv::Point(x1i, y1i), c3d, 2,
                         cv::LINE_AA);
              };
              draw_edge(0, 1);
              draw_edge(1, 2);
              draw_edge(2, 3);
              draw_edge(3, 0);
              draw_edge(4, 5);
              draw_edge(5, 6);
              draw_edge(6, 7);
              draw_edge(7, 4);
              draw_edge(0, 4);
              draw_edge(1, 5);
              draw_edge(2, 6);
              draw_edge(3, 7);
            }
          }

          if (image_raw_publisher_ && !bgr.empty())
          {
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr)
                           .toImageMsg();
            msg->header.stamp.sec = item.nv12_frame.stamp_sec;
            msg->header.stamp.nanosec = item.nv12_frame.stamp_nanosec;
            image_raw_publisher_->publish(*msg);
          }

          if (general_config_.show_gui && !bgr.empty())
          {
            cv::resize(bgr, bgr, cv::Size(bgr.cols * 0.7, bgr.rows * 0.5));
            cv::imshow("Single-Source Display", bgr);
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27)
            {
              running_ = false;
              queue_cv_.notify_all();
              nv12_queue_cv_.notify_all();
              draw_queue_cv_.notify_all();
            }
          }
        }
      }

      if (nitros_sub_pool_)
      {
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
    if (input_config_.ros2_enabled && ros2_client_ && !proc.bboxes.empty() &&
        !proc.bboxes[0].empty() && !fusion_img.empty())
    {
      submitDistanceRequest(item.stamp, proc.bboxes[0]);
      DistanceBundle bundle;
      if (!tryGetDistanceCache(item.stamp, bundle))
      {
        bundle.objects.clear();
      }

      const int line_len_px = 120;
      const int dash_len = 8;
      const int gap_len = 6;
      const int thickness = 1;
      const cv::Scalar horiz_color(0, 255, 255);
      const cv::Scalar vert_color(255, 0, 255);

      for (const auto &m : bundle.objects)
      {
        if (!m.has_distance)
        {
          continue;
        }
        const int cx =
            static_cast<int>(std::lround(static_cast<double>(m.center_u)));
        const int cy =
            static_cast<int>(std::lround(static_cast<double>(m.center_v)));
        if (cx < 0 || cx >= fusion_img.cols || cy < 0 ||
            cy >= fusion_img.rows)
        {
          continue;
        }

        const int x1 = std::min(cx + line_len_px, fusion_img.cols - 1);
        const int y1 = std::min(cy + line_len_px, fusion_img.rows - 1);

        DrawDashedHLine(fusion_img, cx, x1, cy, horiz_color, thickness,
                        dash_len, gap_len);
        DrawDashedVLine(fusion_img, cx, cy, y1, vert_color, thickness, dash_len,
                        gap_len);

        const std::string d_text = cv::format("D=%.2fm", m.depth_m);
        const std::string h_text = cv::format("H=%.2fm", m.height_m);

        const int dx = std::min(x1 + 6, fusion_img.cols - 1);
        const int dy = std::max(cy - 6, 0);
        const int hx = std::min(cx + 6, fusion_img.cols - 1);
        const int hy = std::max(y1 - 6, 0);

        cv::putText(fusion_img, d_text, cv::Point(dx, dy),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, horiz_color, 1, cv::LINE_AA);
        cv::putText(fusion_img, h_text, cv::Point(hx, hy),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, vert_color, 1, cv::LINE_AA);
      }
    }
    auto draw_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> draw_duration = draw_end - draw_start;
    sample::gLogInfo << "Draw time: " << draw_duration.count() * 1000 << "ms"
                     << std::endl;

    if (input_config_.ros2_enabled && ros2_client_)
    {
      if (image_encoder_decoder_config_.encoding_format_pub ==
          EncodingFormat::NITROS_NV12)
      {
        publishNitrosImage(fusion_img);
      }
      else if (image_raw_publisher_)
      {
        auto msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", fusion_img)
                .toImageMsg();
        msg->header.stamp = ros2_client_->getNode()->now();
        image_raw_publisher_->publish(*msg);
      }
    }

    if (general_config_.show_gui && !fusion_img.empty())
    {
      cv::resize(fusion_img, fusion_img,
                 cv::Size(fusion_img.cols * 0.7, fusion_img.rows * 0.5));
      cv::imshow("Single-Source Display", fusion_img);
      char key = static_cast<char>(cv::waitKey(1));
      if (key == 27)
      {
        running_ = false;
        queue_cv_.notify_all();
        nv12_queue_cv_.notify_all();
        draw_queue_cv_.notify_all();
      }
    }
  }
}

void CameraCombinedWithLidarPipelineManager::detectionThread()
{
  while (running_)
  {
    const bool skip_yolo_detect =
        (input_config_.ros2_enabled && use_nitros_sub_);
    if (input_config_.ros2_enabled) // ros2模式下读取相机图像
    {
      if (use_nitros_sub_)
      {
        Nv12GpuFrame nv12_frame;
        {
          std::unique_lock<std::mutex> lock(nv12_queue_mutex_);
          nv12_queue_cv_.wait(
              lock, [this]
              { return !nv12_queue_.empty() || !running_; });

          if (!running_ && nv12_queue_.empty())
          {
            return;
          }

          while (nv12_queue_.size() > 1)
          {
            auto old = nv12_queue_.front();
            nv12_queue_.pop();
            if (nitros_sub_pool_)
            {
              nitros_sub_pool_->release(old.pool_index);
            }
          }
          nv12_frame = nv12_queue_.front();
          nv12_queue_.pop();
        }

        if (!nv12_frame.gpu_buffer || nv12_frame.width <= 0 ||
            nv12_frame.height <= 0)
        {
          if (nitros_sub_pool_)
          {
            nitros_sub_pool_->release(nv12_frame.pool_index);
          }
          continue;
        }

        std::vector<std::vector<Box>> bboxes;
        bboxes.resize(1);

        const bool detection_enabled =
            IsDetectionEnabledByMqtt(mqtt_publisher_, mqtt_publisher_mutex_);
        if (!detection_enabled)
        {
          ResetMqttAlarmsIfNeeded(tooth_detection_config_,
                                  pedestrian_detection_config_, mqtt_publisher_,
                                  mqtt_publisher_mutex_);

          tooth_detection_config_.tooth_state_name.clear();
          tooth_detection_config_.teeth_root_coordinates.clear();
          tooth_detection_config_.root_idx.clear();
          pedestrian_detection_config_.people_cnt = 0;
          pedestrian_detection_config_.show_people_cnt = 0;

          image_processing_config_.bboxes = bboxes;
        }
        else
        {
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
          if (!objectss.empty())
          {
            primary_boxes = objectss[0];
          }

          const auto &names = yolo_->getParam().class_names;
          int people_idx = -1;
          for (size_t i = 0; i < names.size(); ++i)
          {
            if (names[i] == "people")
            {
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

          if (do_people_second_classify)
          {
            std::vector<Box> people_candidates;
            people_candidates.reserve(primary_boxes.size());
            for (const auto &b : primary_boxes)
            {
              if (b.label == people_idx)
              {
                people_candidates.push_back(b);
              }
              else
              {
                non_people_boxes.push_back(b);
              }
            }

            people_boxes.reserve(people_candidates.size());
            for (auto &pb : people_candidates)
            {
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
              if (!classifier_output.empty() && classifier_output[0] > 0.5f)
              {
                people_boxes.push_back(pb);
                people_cnt += 1;
              }
            }
          }
          else
          {
            constexpr float kPeopleConfidenceThreshold = 0.7f;
            SplitPrimaryBoxesByPeopleThreshold(
                primary_boxes, people_idx, kPeopleConfidenceThreshold,
                people_boxes, non_people_boxes, people_cnt);
          }

          pedestrian_detection_config_.people_cnt = people_cnt;
          pedestrian_detection_config_.show_people_cnt = people_cnt;

          std::vector<Box> final_boxes;
          if (model_config_.need_second_detection_ && second_yolo_)
          {
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
          }
          else
          {
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
          const rclcpp::Time stamp(nv12_frame.stamp_sec,
                                   nv12_frame.stamp_nanosec, RCL_ROS_TIME);
          if (!image_processing_config_.bboxes.empty())
          {
            const auto &boxes0 = image_processing_config_.bboxes[0];
            submitDistanceRequest(stamp, boxes0);
          }
          else
          {
            submitDistanceRequest(stamp, {});
          }
        }

        {
          std::lock_guard<std::mutex> last_lock(last_detection_mutex_);
          last_detection_boxes_ = image_processing_config_.bboxes;
        }

        if (nitros_sub_pool_)
        {
          nitros_sub_pool_->release(nv12_frame.pool_index);
        }
        continue;
      }
      else
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock,
                       [this]
                       { return !image_queue_.empty() || !running_; });

        if (!running_ && image_queue_.empty())
          return;

        while (image_queue_.size() > 1)
        {
          image_queue_.pop();
        }
        image_processing_config_.init_frame =
            image_queue_.front(); // 只取最新的图像
        image_queue_.pop();
      }
    }
    else // 非ros2模式下读取帧
    {
      auto open_start = std::chrono::high_resolution_clock::now();
      image_processing_config_.captures[input_config_.idx].read(
          image_processing_config_.init_frame);
      if (image_processing_config_.init_frame.empty())
      {
        running_ = false;
        break; // 立即退出当前线程的循环
      }
      // V4L2 常输出 YUYV/YUY2（CV_8UC2）。后续推理/绘制按 BGR(3通道)
      // 使用会越界崩溃。
      if (image_processing_config_.init_frame.channels() != 3)
      {
        if (image_processing_config_.init_frame.type() == CV_8UC2)
        {
          cv::Mat bgr;
          try
          {
            cv::cvtColor(image_processing_config_.init_frame, bgr,
                         cv::COLOR_YUV2BGR_YUY2);
          }
          catch (const cv::Exception &)
          {
            running_ = false;
            break;
          }
          image_processing_config_.init_frame = std::move(bgr);
        }
        else if (image_processing_config_.init_frame.type() == CV_8UC4)
        {
          cv::Mat bgr;
          try
          {
            cv::cvtColor(image_processing_config_.init_frame, bgr,
                         cv::COLOR_BGRA2BGR);
          }
          catch (const cv::Exception &)
          {
            running_ = false;
            break;
          }
          image_processing_config_.init_frame = std::move(bgr);
        }
        else
        {
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

    if (!skip_yolo_detect)
    {
      auto detect_start = std::chrono::high_resolution_clock::now();
      if (model_config_.need_frame_skip_)
      {
        bool should_detect = process_frame_ % num_sources == 0;
        if (should_detect)
        {
          yolo_detect::detect(
              yolo_, second_yolo_, trt_classifier_, undistorter_,
              mqtt_publisher_, ros2_client_, model_config_, undistort_config_,
              input_config_, tooth_detection_config_,
              pedestrian_detection_config_, image_processing_config_,
              general_config_, roi_config_, mqtt_config_);

          previous_bboxes_ =
              image_processing_config_.bboxes; // 缓存当前检测结果
        }
        else
        {
          image_processing_config_.bboxes =
              previous_bboxes_; // 使用上一次的检测结果
        }
        process_frame_ = (process_frame_ + 1) % num_sources; // 更新处理帧计数
      }
      else
      {
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

    rclcpp::Time distance_stamp{0, 0, RCL_ROS_TIME};
    if (input_config_.ros2_enabled && ros2_client_)
    {
      distance_stamp = ros2_client_->getNode()->now();
      if (!image_processing_config_.bboxes.empty())
      {
        computeAndPublishDistances(distance_stamp,
                                   image_processing_config_.bboxes[0]);
      }
      else
      {
        computeAndPublishDistances(distance_stamp, {});
      }
    }

    {
      DrawItem item;
      item.is_nv12 = false;
      item.bgr_frame = std::move(image_processing_config_.init_frame);
      item.bboxes = image_processing_config_.bboxes;
      item.stamp = distance_stamp;
      item.tooth_detection_config =
          std::make_shared<ToothDetectionConfig>(tooth_detection_config_);
      item.pedestrian_detection_config =
          std::make_shared<PedestrianDetectionConfig>(
              pedestrian_detection_config_);

      std::lock_guard<std::mutex> lk(draw_queue_mutex_);
      while (draw_queue_.size() >= max_draw_queue_size_)
      {
        DrawItem old = std::move(draw_queue_.front());
        draw_queue_.pop();
        if (old.is_nv12 && nitros_sub_pool_)
        {
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

void CameraCombinedWithLidarPipelineManager::start()
{
  // 启动检测线程
  detection_thread_ = std::thread(
      &CameraCombinedWithLidarPipelineManager::detectionThread, this);
  draw_thread_ =
      std::thread(&CameraCombinedWithLidarPipelineManager::drawThread, this);

  if (input_config_.ros2_enabled)
  {
    if (!distance_thread_.joinable())
    {
      distance_thread_ = std::thread(
          &CameraCombinedWithLidarPipelineManager::distanceWorker, this);
    }
    if (parent_node_)
    {
      return;
    }
    ros_spin_thread_ = std::thread([this]()
                                   {
      while (running_) {
        rclcpp::spin_some(ros2_client_->get_node_base_interface());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      } });
  }

  if (detection_thread_.joinable())
  {
    detection_thread_.join();
  }
  if (draw_thread_.joinable())
  {
    draw_thread_.join();
  }
  running_ = false;
  queue_cv_.notify_all();
  nv12_queue_cv_.notify_all();
  draw_queue_cv_.notify_all();
  distance_req_cond_.notify_all();
  if (distance_thread_.joinable())
  {
    distance_thread_.join();
  }
  if (ros_spin_thread_.joinable())
  {
    ros_spin_thread_.join();
  }
}

void CameraCombinedWithLidarPipelineManager::stop()
{
  running_ = false;
  queue_cv_.notify_all();
  nv12_queue_cv_.notify_all();
  draw_queue_cv_.notify_all();
  distance_req_cond_.notify_all();
  if (detection_thread_.joinable())
  {
    detection_thread_.join();
  }
  if (draw_thread_.joinable())
  {
    draw_thread_.join();
  }
  if (ros_spin_thread_.joinable())
  {
    ros_spin_thread_.join();
  }
  if (distance_thread_.joinable())
  {
    distance_thread_.join();
  }
  cleanupResources();
}
