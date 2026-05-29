#include <cv_bridge/cv_bridge.h>
#include <cuda_runtime.h>
#include <isaac_ros_managed_nitros/managed_nitros_publisher.hpp>
#include <isaac_ros_managed_nitros/managed_nitros_subscriber.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <isaac_ros_nitros_image_type/nitros_image_view.hpp>
#if __has_include(<livox_ros_driver2/msg/custom_msg.hpp>)
#include <livox_ros_driver2/msg/custom_msg.hpp>
#define ISAAC_ROS_CAM_AND_LIDAR_HAS_LIVOX_CUSTOM_MSG 1
#else
#define ISAAC_ROS_CAM_AND_LIDAR_HAS_LIVOX_CUSTOM_MSG 0
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../isaac_ros_object_detection/src/perception/gxf_nv12_convert.h"
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

std::string Trim(std::string value)
{
  const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!value.empty() && is_space(static_cast<unsigned char>(value.front())))
  {
    value.erase(value.begin());
  }
  while (!value.empty() && is_space(static_cast<unsigned char>(value.back())))
  {
    value.pop_back();
  }
  return value;
}

void ExtractDoubles(const std::string &text, std::vector<double> &values)
{
  const char *ptr = text.c_str();
  while (*ptr != '\0')
  {
    char *end = nullptr;
    const double number = std::strtod(ptr, &end);
    if (end != ptr)
    {
      values.push_back(number);
      ptr = end;
    }
    else
    {
      ++ptr;
    }
  }
}

double Clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(max_value, value));
}

struct Nv12PublishPool
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

  size_t size_bytes{0};
  std::mutex m;
  std::vector<void *> free_list;
};

static nvidia::isaac_ros::nitros::NitrosImage BuildNv12NitrosImageWithPool(
  const std_msgs::msg::Header &header,
  uint32_t height,
  uint32_t width,
  void *gpu_data,
  int y_stride,
  int uv_stride,
  const std::shared_ptr<Nv12PublishPool> &pool)
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

  auto gxf_image = message->add<nvidia::gxf::VideoBuffer>(header.frame_id.c_str());
  if (!gxf_image)
  {
    std::stringstream error_msg;
    error_msg << "[BuildNv12NitrosImageWithPool] Failed to add VideoBuffer: "
              << GxfResultStr(gxf_image.error());
    throw std::runtime_error(error_msg.str().c_str());
  }

  if ((width % 2) != 0 || (height % 2) != 0)
  {
    throw std::runtime_error(
            "[BuildNv12NitrosImageWithPool] width/height must be even for NV12");
  }

  using GxfVideoFormat = nvidia::gxf::VideoFormat;
  nvidia::gxf::VideoFormatSize<GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER> format_size;
  std::array<nvidia::gxf::ColorPlane, 2> planes{
    nvidia::gxf::ColorPlane("Y", 1, static_cast<uint32_t>(y_stride)),
    nvidia::gxf::ColorPlane("UV", 2, static_cast<uint32_t>(uv_stride)),
  };
  const uint64_t size = format_size.size(width, height, planes, false);
  std::vector<nvidia::gxf::ColorPlane> color_planes{planes.begin(), planes.end()};

  constexpr auto surface_layout =
    nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  constexpr auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;

  nvidia::gxf::VideoBufferInfo buffer_info{
    width,
    height,
    GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER,
    std::move(color_planes),
    surface_layout};

  gxf_image.value()->wrapMemory(
    buffer_info, size, storage_type, gpu_data,
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
}  // namespace

class CameraLidarProjectionDemo : public rclcpp::Node
{
public:
  explicit CameraLidarProjectionDemo(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
  : Node("camera_lidar_projection_demo", options)
  {
    image_topic_ = declare_parameter<std::string>("image_topic", "/front/image_raw");
    use_nitros_image_ = declare_parameter<bool>("use_nitros_image", true);
    nitros_image_format_ =
      declare_parameter<std::string>("nitros_image_format", "nitros_image_nv12");
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/livox/lidar");
    use_livox_custom_msg_ =
      declare_parameter<bool>("use_livox_custom_msg", true);
#if !ISAAC_ROS_CAM_AND_LIDAR_HAS_LIVOX_CUSTOM_MSG
    if (use_livox_custom_msg_)
    {
      RCLCPP_WARN(
        get_logger(),
        "livox_ros_driver2 is not available at build time. "
        "Falling back to sensor_msgs/msg/PointCloud2 subscription.");
      use_livox_custom_msg_ = false;
    }
#endif
    overlay_topic_ =
      declare_parameter<std::string>("overlay_topic", "/front/projection_overlay");
    bev_topic_ =
      declare_parameter<std::string>("bev_topic", "/front/projection_bev");
    colored_cloud_topic_ =
      declare_parameter<std::string>("colored_cloud_topic", "/front/projection_colored_cloud");
    publish_colored_cloud_ = declare_parameter<bool>("publish_colored_cloud", true);
    enable_colored_cloud_sliding_window_ =
      declare_parameter<bool>("enable_colored_cloud_sliding_window", true);
    const int colored_cloud_window_ms =
      declare_parameter<int>("colored_cloud_window_ms", 1000);
    const int colored_cloud_max_points =
      declare_parameter<int>("colored_cloud_max_points", 200000);
    draw_bev_on_image_callback_ = declare_parameter<bool>("draw_bev_on_image_callback", false);
    calib_result_path_ = declare_parameter<std::string>(
      "calib_result_path",
      "/home/alienware/Sensor/ISAAC/src/isaac_ros_object_detection/config/calib_result.yaml");
    bev_width_ = declare_parameter<int>("bev_width", 800);
    bev_height_ = declare_parameter<int>("bev_height", 1200);
    bev_x_min_m_ = declare_parameter<double>("bev_x_min_m", 0.0);
    bev_x_max_m_ = declare_parameter<double>("bev_x_max_m", 15.0);
    bev_y_min_m_ = declare_parameter<double>("bev_y_min_m", -5.0);
    bev_y_max_m_ = declare_parameter<double>("bev_y_max_m", 5.0);
    bev_flip_y_ = declare_parameter<bool>("bev_flip_y", true);
    const int draw_every_nth_point =
      declare_parameter<int>("draw_every_nth_point", 2);
    const int point_radius_px =
      declare_parameter<int>("point_radius_px", 2);
    const int camera_frame_timeout_ms =
      declare_parameter<int>("camera_frame_timeout_ms", 300);
    const int cloud_frame_timeout_ms =
      declare_parameter<int>("cloud_frame_timeout_ms", 300);
    const double grid_step_m =
      declare_parameter<double>("grid_step_m", 1.0);
    enable_cloud_filter_ = declare_parameter<bool>("enable_cloud_filter", true);
    const double filter_min_range_m =
      declare_parameter<double>("filter_min_range_m", 0.5);
    const double filter_max_range_m =
      declare_parameter<double>("filter_max_range_m", 80.0);
    const double filter_min_z_m =
      declare_parameter<double>("filter_min_z_m", -2.5);
    const double filter_max_z_m =
      declare_parameter<double>("filter_max_z_m", 3.0);
    const double filter_voxel_leaf_size_m =
      declare_parameter<double>("filter_voxel_leaf_size_m", 0.08);
    const double filter_radius_m =
      declare_parameter<double>("filter_radius_m", 0.25);
    const int filter_min_neighbors =
      declare_parameter<int>("filter_min_neighbors", 2);
    input_nv12_variant_ =
      declare_parameter<std::string>("input_nv12_variant", "nv21");

    draw_every_nth_point_ = std::max<int>(1, draw_every_nth_point);
    point_radius_px_ = std::max<int>(1, point_radius_px);
    camera_frame_timeout_ms_ = std::max<int>(1, camera_frame_timeout_ms);
    cloud_frame_timeout_ms_ = std::max<int>(1, cloud_frame_timeout_ms);
    colored_cloud_window_ms_ = std::max<int>(1, colored_cloud_window_ms);
    colored_cloud_max_points_ = std::max<int>(1, colored_cloud_max_points);
    grid_step_m_ = std::max<double>(0.1, grid_step_m);
    filter_min_range_m_ = std::max<double>(0.0, filter_min_range_m);
    filter_max_range_m_ =
      std::max<double>(filter_min_range_m_ + 0.1, filter_max_range_m);
    if (filter_min_z_m <= filter_max_z_m)
    {
      filter_min_z_m_ = filter_min_z_m;
      filter_max_z_m_ = filter_max_z_m;
    }
    else
    {
      filter_min_z_m_ = filter_max_z_m;
      filter_max_z_m_ = filter_min_z_m;
    }
    filter_voxel_leaf_size_m_ = std::max<double>(0.0, filter_voxel_leaf_size_m);
    filter_radius_m_ = std::max<double>(0.0, filter_radius_m);
    filter_min_neighbors_ = std::max<int>(0, filter_min_neighbors);
    overlay_only_front_points_ =
      declare_parameter<bool>("overlay_only_front_points", true);

    if (!LoadCalibration())
    {
      RCLCPP_FATAL(get_logger(), "Failed to load calibration from %s",
                   calib_result_path_.c_str());
      throw std::runtime_error("calibration load failed");
    }

    nitros_overlay_pub_ = std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>>(
      this, overlay_topic_,
      nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name);
    bev_pub_ = create_publisher<sensor_msgs::msg::Image>(bev_topic_, 10);
    colored_cloud_pub_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
        colored_cloud_topic_, rclcpp::QoS(10).reliable());

    image_callback_group_ =
      create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    lidar_callback_group_ =
      create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions image_sub_options;
    image_sub_options.callback_group = image_callback_group_;
    rclcpp::SubscriptionOptions lidar_sub_options;
    lidar_sub_options.callback_group = lidar_callback_group_;

    if (use_nitros_image_)
    {
      nitros_image_sub_ = std::make_shared<
        nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
          nvidia::isaac_ros::nitros::NitrosImageView>>(
        this, image_topic_, nitros_image_format_,
        std::function<void(const nvidia::isaac_ros::nitros::NitrosImageView &)>(
          [this](const nvidia::isaac_ros::nitros::NitrosImageView &view)
          {
            this->OnNitrosImage(view);
          }),
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
        rclcpp::SensorDataQoS());
    }
    else
    {
      image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&CameraLidarProjectionDemo::OnImage, this, std::placeholders::_1),
        image_sub_options);
    }

    if (use_livox_custom_msg_)
    {
#if ISAAC_ROS_CAM_AND_LIDAR_HAS_LIVOX_CUSTOM_MSG
      livox_sub_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
        lidar_topic_, rclcpp::SensorDataQoS(),
        std::bind(&CameraLidarProjectionDemo::OnLivoxCustomMsg, this, std::placeholders::_1),
        lidar_sub_options);
#endif
    }
    else
    {
      pointcloud2_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, rclcpp::SensorDataQoS(),
        std::bind(&CameraLidarProjectionDemo::OnPointCloud2, this, std::placeholders::_1),
        lidar_sub_options);
    }

  }

  ~CameraLidarProjectionDemo() override
  {
    if (cuda_input_copy_buffer_ != nullptr)
    {
      cudaFree(cuda_input_copy_buffer_);
      cuda_input_copy_buffer_ = nullptr;
      cuda_input_copy_buffer_size_ = 0;
    }
    if (cuda_bgr_publish_buffer_ != nullptr)
    {
      cudaFree(cuda_bgr_publish_buffer_);
      cuda_bgr_publish_buffer_ = nullptr;
      cuda_bgr_publish_buffer_size_ = 0;
    }
    if (cuda_stream_ != nullptr)
    {
      cudaStreamDestroy(cuda_stream_);
      cuda_stream_ = nullptr;
    }
  }

private:
  struct CameraModel
  {
    enum class Type
    {
      kPinholeRational,
      kFisheye
    };

    bool valid{false};
    Type type{Type::kPinholeRational};
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    std::array<double, 8> dist_pinhole{};
    std::array<double, 4> dist_fisheye{};
    std::array<double, 9> r_cl{};
    std::array<double, 3> t_cl{};
  };

  struct LidarPoint
  {
    cv::Vec3f lidar;
    cv::Vec3f camera;
    cv::Point2f uv;
    bool projected{false};
  };

  struct GridKey
  {
    int x{0};
    int y{0};
    int z{0};

    bool operator==(const GridKey &other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct GridKeyHash
  {
    size_t operator()(const GridKey &key) const noexcept
    {
      size_t seed = std::hash<int>{}(key.x);
      seed ^= std::hash<int>{}(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= std::hash<int>{}(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };

  struct VoxelAccumulator
  {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    size_t count{0};
  };

  struct ColoredPoint
  {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    uint32_t rgb{0};
  };

  struct ColoredCloudBatch
  {
    rclcpp::Time stamp;
    std::vector<ColoredPoint> points;
  };

  bool LoadCalibration()
  {
    std::ifstream ifs(calib_result_path_);
    if (!ifs.good())
    {
      RCLCPP_ERROR(get_logger(), "Cannot open calibration file: %s",
                   calib_result_path_.c_str());
      return false;
    }

    CameraModel model;
    std::string line;
    std::string model_str;
    bool have_fx = false;
    bool have_fy = false;
    bool have_cx = false;
    bool have_cy = false;
    bool have_r = false;
    bool have_t = false;
    bool reading_r = false;
    bool reading_t = false;
    std::vector<double> r_values;
    std::vector<double> t_values;
    std::string section;

    auto parse_key_double = [&](const std::string &prefix, double &out, bool &have) {
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
        out = std::stod(Trim(line.substr(pos + 1)));
        have = true;
      }
      catch (...)
      {
      }
    };

    while (std::getline(ifs, line))
    {
      line = Trim(line);
      if (line.empty() || line[0] == '#')
      {
        continue;
      }

      const auto colon = line.find(':');
      if (colon != std::string::npos)
      {
        const std::string key = Trim(line.substr(0, colon));
        const std::string value = Trim(line.substr(colon + 1));
        if (!key.empty() && value.empty())
        {
          section = key;
          std::transform(section.begin(), section.end(), section.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
          });
        }
        if (key == "model" || key == "cam_model")
        {
          model_str = value;
        }
      }

      parse_key_double("cam_fx", model.fx, have_fx);
      parse_key_double("cam_fy", model.fy, have_fy);
      parse_key_double("cam_cx", model.cx, have_cx);
      parse_key_double("cam_cy", model.cy, have_cy);
      parse_key_double("fx", model.fx, have_fx);
      parse_key_double("fy", model.fy, have_fy);
      parse_key_double("cx", model.cx, have_cx);
      parse_key_double("cy", model.cy, have_cy);

      if (section == "fisheye" || section == "fish_eye" || section == "fish-eye")
      {
        bool dummy = false;
        parse_key_double("k1", model.dist_fisheye[0], dummy);
        parse_key_double("k2", model.dist_fisheye[1], dummy);
        parse_key_double("k3", model.dist_fisheye[2], dummy);
        parse_key_double("k4", model.dist_fisheye[3], dummy);
      }
      else
      {
        bool dummy = false;
        parse_key_double("k1", model.dist_pinhole[0], dummy);
        parse_key_double("k2", model.dist_pinhole[1], dummy);
        parse_key_double("p1", model.dist_pinhole[2], dummy);
        parse_key_double("p2", model.dist_pinhole[3], dummy);
        parse_key_double("k3", model.dist_pinhole[4], dummy);
        parse_key_double("k4", model.dist_pinhole[5], dummy);
        parse_key_double("k5", model.dist_pinhole[6], dummy);
        parse_key_double("k6", model.dist_pinhole[7], dummy);
        parse_key_double("cam_d0", model.dist_pinhole[0], dummy);
        parse_key_double("cam_d1", model.dist_pinhole[1], dummy);
        parse_key_double("cam_d2", model.dist_pinhole[2], dummy);
        parse_key_double("cam_d3", model.dist_pinhole[3], dummy);
      }

      const bool is_r_line =
        (line.rfind("Rcl", 0) == 0) || (line.rfind("R_cl", 0) == 0) ||
        (line.rfind("R:", 0) == 0) || (line.rfind("R =", 0) == 0) ||
        (line.rfind("R=", 0) == 0);
      const bool is_t_line =
        (line.rfind("Pcl", 0) == 0) || (line.rfind("tcl", 0) == 0) ||
        (line.rfind("t:", 0) == 0) || (line.rfind("t =", 0) == 0) ||
        (line.rfind("t=", 0) == 0) || (line.rfind("P:", 0) == 0) ||
        (line.rfind("P =", 0) == 0) || (line.rfind("P=", 0) == 0);

      if (is_r_line)
      {
        reading_r = true;
        r_values.clear();
      }
      if (is_t_line)
      {
        reading_t = true;
        t_values.clear();
      }

      if (reading_r && !have_r)
      {
        ExtractDoubles(line, r_values);
        if (r_values.size() >= 9)
        {
          for (size_t i = 0; i < 9; ++i)
          {
            model.r_cl[i] = r_values[i];
          }
          have_r = true;
          reading_r = false;
        }
      }

      if (reading_t && !have_t)
      {
        ExtractDoubles(line, t_values);
        if (t_values.size() >= 3)
        {
          for (size_t i = 0; i < 3; ++i)
          {
            model.t_cl[i] = t_values[i];
          }
          have_t = true;
          reading_t = false;
        }
      }
    }

    std::string normalized_model = Trim(model_str);
    std::transform(
      normalized_model.begin(), normalized_model.end(), normalized_model.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized_model.find("fish") != std::string::npos)
    {
      model.type = CameraModel::Type::kFisheye;
    }

    model.valid = have_fx && have_fy && have_cx && have_cy && have_r && have_t;
    if (!model.valid)
    {
      RCLCPP_ERROR(get_logger(), "Calibration file is missing required intrinsics or extrinsics");
      return false;
    }

    camera_model_ = model;
    return true;
  }

  bool ProjectToImage(const cv::Vec3f &camera_point, cv::Point2f &uv) const
  {
    const double x = static_cast<double>(camera_point[0]);
    const double y = static_cast<double>(camera_point[1]);
    const double z = static_cast<double>(camera_point[2]);
    if (!(z > 1e-6))
    {
      return false;
    }

    const double xn = x / z;
    const double yn = y / z;

    if (camera_model_.type == CameraModel::Type::kFisheye)
    {
      const double r = std::sqrt(xn * xn + yn * yn);
      double xd = xn;
      double yd = yn;
      if (r > 1e-12)
      {
        const double theta = std::atan(r);
        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta4 * theta4;
        const auto &d = camera_model_.dist_fisheye;
        const double theta_d = theta * (1.0 + d[0] * theta2 + d[1] * theta4 +
                                        d[2] * theta6 + d[3] * theta8);
        const double scale = theta_d / r;
        xd = xn * scale;
        yd = yn * scale;
      }
      uv.x = static_cast<float>(camera_model_.fx * xd + camera_model_.cx);
      uv.y = static_cast<float>(camera_model_.fy * yd + camera_model_.cy);
      return std::isfinite(uv.x) && std::isfinite(uv.y);
    }

    const auto &d = camera_model_.dist_pinhole;
    const double r2 = xn * xn + yn * yn;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    double radial = 1.0 + d[0] * r2 + d[1] * r4 + d[4] * r6;
    const double denom = 1.0 + d[5] * r2 + d[6] * r4 + d[7] * r6;
    if (std::fabs(denom) > 1e-12)
    {
      radial /= denom;
    }
    const double x2 = xn * xn;
    const double y2 = yn * yn;
    const double xy = xn * yn;
    const double x_tang = 2.0 * d[2] * xy + d[3] * (r2 + 2.0 * x2);
    const double y_tang = d[2] * (r2 + 2.0 * y2) + 2.0 * d[3] * xy;
    const double xd = xn * radial + x_tang;
    const double yd = yn * radial + y_tang;
    uv.x = static_cast<float>(camera_model_.fx * xd + camera_model_.cx);
    uv.y = static_cast<float>(camera_model_.fy * yd + camera_model_.cy);
    return std::isfinite(uv.x) && std::isfinite(uv.y);
  }

  cv::Vec3f TransformLidarToCamera(float x, float y, float z) const
  {
    const auto &r = camera_model_.r_cl;
    const auto &t = camera_model_.t_cl;
    return cv::Vec3f(
      static_cast<float>(r[0] * x + r[1] * y + r[2] * z + t[0]),
      static_cast<float>(r[3] * x + r[4] * y + r[5] * z + t[1]),
      static_cast<float>(r[6] * x + r[7] * y + r[8] * z + t[2]));
  }

  GridKey MakeGridKey(const cv::Vec3f &point, double cell_size) const
  {
    const double inv = 1.0 / cell_size;
    return GridKey{
      static_cast<int>(std::floor(static_cast<double>(point[0]) * inv)),
      static_cast<int>(std::floor(static_cast<double>(point[1]) * inv)),
      static_cast<int>(std::floor(static_cast<double>(point[2]) * inv))};
  }

  std::vector<cv::Vec3f> ApplyVoxelGridFilter(const std::vector<cv::Vec3f> &points) const
  {
    if (points.empty() || filter_voxel_leaf_size_m_ <= 1e-6)
    {
      return points;
    }

    std::unordered_map<GridKey, VoxelAccumulator, GridKeyHash> voxels;
    voxels.reserve(points.size());
    for (const auto &point : points)
    {
      auto &acc = voxels[MakeGridKey(point, filter_voxel_leaf_size_m_)];
      acc.x += static_cast<double>(point[0]);
      acc.y += static_cast<double>(point[1]);
      acc.z += static_cast<double>(point[2]);
      ++acc.count;
    }

    std::vector<cv::Vec3f> downsampled;
    downsampled.reserve(voxels.size());
    for (const auto &entry : voxels)
    {
      const auto &acc = entry.second;
      if (acc.count == 0)
      {
        continue;
      }
      const double scale = 1.0 / static_cast<double>(acc.count);
      downsampled.emplace_back(
        static_cast<float>(acc.x * scale),
        static_cast<float>(acc.y * scale),
        static_cast<float>(acc.z * scale));
    }

    return downsampled;
  }

  std::vector<cv::Vec3f> ApplyRadiusOutlierFilter(const std::vector<cv::Vec3f> &points) const
  {
    if (points.empty() || filter_radius_m_ <= 1e-6 || filter_min_neighbors_ <= 0)
    {
      return points;
    }
    if (points.size() <= static_cast<size_t>(filter_min_neighbors_))
    {
      return points;
    }

    std::unordered_map<GridKey, std::vector<size_t>, GridKeyHash> buckets;
    buckets.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
      buckets[MakeGridKey(points[i], filter_radius_m_)].push_back(i);
    }

    const float radius2 = static_cast<float>(filter_radius_m_ * filter_radius_m_);
    std::vector<cv::Vec3f> filtered;
    filtered.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
      const GridKey center = MakeGridKey(points[i], filter_radius_m_);
      int neighbor_count = 0;
      bool keep_point = false;
      for (int dx = -1; dx <= 1 && !keep_point; ++dx)
      {
        for (int dy = -1; dy <= 1 && !keep_point; ++dy)
        {
          for (int dz = -1; dz <= 1 && !keep_point; ++dz)
          {
            const auto it = buckets.find(GridKey{center.x + dx, center.y + dy, center.z + dz});
            if (it == buckets.end())
            {
              continue;
            }
            for (const size_t neighbor_index : it->second)
            {
              if (neighbor_index == i)
              {
                continue;
              }
              const cv::Vec3f diff = points[neighbor_index] - points[i];
              if (diff.dot(diff) <= radius2)
              {
                ++neighbor_count;
                if (neighbor_count >= filter_min_neighbors_)
                {
                  filtered.push_back(points[i]);
                  keep_point = true;
                  break;
                }
              }
            }
          }
        }
      }
    }

    return filtered;
  }

  std::vector<cv::Vec3f> FilterRawPointCloud(const std::vector<cv::Vec3f> &raw_points) const
  {
    if (!enable_cloud_filter_)
    {
      return raw_points;
    }

    std::vector<cv::Vec3f> pass_through;
    pass_through.reserve(raw_points.size());
    const double min_range2 = filter_min_range_m_ * filter_min_range_m_;
    const double max_range2 = filter_max_range_m_ * filter_max_range_m_;
    for (const auto &point : raw_points)
    {
      const double x = static_cast<double>(point[0]);
      const double y = static_cast<double>(point[1]);
      const double z = static_cast<double>(point[2]);
      const double range2 = x * x + y * y + z * z;
      if (range2 < min_range2 || range2 > max_range2)
      {
        continue;
      }
      if (z < filter_min_z_m_ || z > filter_max_z_m_)
      {
        continue;
      }
      pass_through.push_back(point);
    }

    auto voxel_filtered = ApplyVoxelGridFilter(pass_through);
    return ApplyRadiusOutlierFilter(voxel_filtered);
  }

  std::vector<LidarPoint> BuildFilteredLidarPoints(const std::vector<cv::Vec3f> &raw_points) const
  {
    const auto filtered_points = FilterRawPointCloud(raw_points);

    std::vector<LidarPoint> points;
    points.reserve(filtered_points.size());
    for (const auto &lidar : filtered_points)
    {
      const cv::Vec3f camera = TransformLidarToCamera(lidar[0], lidar[1], lidar[2]);
      cv::Point2f uv;
      const bool projected = ProjectToImage(camera, uv);
      points.push_back(LidarPoint{lidar, camera, uv, projected});
    }
    return points;
  }

#if ISAAC_ROS_CAM_AND_LIDAR_HAS_LIVOX_CUSTOM_MSG
  void OnLivoxCustomMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    std::vector<cv::Vec3f> raw_points;
    raw_points.reserve(msg->points.size());
    for (const auto &point : msg->points)
    {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
      {
        continue;
      }
      raw_points.emplace_back(point.x, point.y, point.z);
    }
    auto points = BuildFilteredLidarPoints(raw_points);

    const rclcpp::Time stamp(msg->header.stamp);
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      last_cloud_points_ = std::move(points);
      last_cloud_stamp_ = stamp;
      last_cloud_frame_id_ = msg->header.frame_id.empty() ? "livox_frame" : msg->header.frame_id;
    }
    PublishColoredCloudFromLatestImage(stamp);
  }
#endif

  void OnPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    std::vector<cv::Vec3f> raw_points;
    raw_points.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height));
    try
    {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
      {
        const float x = *iter_x;
        const float y = *iter_y;
        const float z = *iter_z;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
          continue;
        }
        raw_points.emplace_back(x, y, z);
      }
    }
    catch (...)
    {
      RCLCPP_WARN(get_logger(), "Failed to parse PointCloud2 message");
      return;
    }
    auto points = BuildFilteredLidarPoints(raw_points);

    const rclcpp::Time stamp(msg->header.stamp);
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      last_cloud_points_ = std::move(points);
      last_cloud_stamp_ = stamp;
      last_cloud_frame_id_ = msg->header.frame_id.empty() ? "livox_frame" : msg->header.frame_id;
    }
    PublishColoredCloudFromLatestImage(stamp);
  }

  void OnImage(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    const int width = static_cast<int>(msg->width);
    const int height = static_cast<int>(msg->height);
    if (width <= 0 || height <= 0 || msg->data.empty())
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Invalid image message: width=%d height=%d data_size=%zu",
        width, height, msg->data.size());
      return;
    }

    cv::Mat bgr;
    try
    {
      if (msg->encoding == sensor_msgs::image_encodings::BGR8)
      {
        bgr = cv::Mat(
          height, width, CV_8UC3, const_cast<unsigned char *>(msg->data.data()),
          msg->step).clone();
      }
      else if (msg->encoding == sensor_msgs::image_encodings::RGB8)
      {
        cv::Mat rgb(
          height, width, CV_8UC3, const_cast<unsigned char *>(msg->data.data()),
          msg->step);
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      }
      else if (msg->encoding == sensor_msgs::image_encodings::BGRA8)
      {
        cv::Mat bgra(
          height, width, CV_8UC4, const_cast<unsigned char *>(msg->data.data()),
          msg->step);
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
      }
      else if (msg->encoding == sensor_msgs::image_encodings::RGBA8)
      {
        cv::Mat rgba(
          height, width, CV_8UC4, const_cast<unsigned char *>(msg->data.data()),
          msg->step);
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
      }
      else if (
        msg->encoding == "yuv422_yuy2" ||
        msg->encoding == "yuyv" ||
        msg->encoding == "yuy2")
      {
        cv::Mat yuy2(
          height, width, CV_8UC2, const_cast<unsigned char *>(msg->data.data()),
          msg->step);
        cv::cvtColor(yuy2, bgr, cv::COLOR_YUV2BGR_YUY2);
      }
      else if (msg->encoding == "nv12")
      {
        const int y_stride = static_cast<int>(msg->step);
        const int uv_stride = y_stride;
        if (y_stride < width || (height % 2) != 0)
        {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Unsupported sensor_msgs NV12 layout: width=%d height=%d step=%d",
            width, height, y_stride);
          return;
        }

        const size_t y_size =
          static_cast<size_t>(y_stride) * static_cast<size_t>(height);
        const size_t uv_size =
          static_cast<size_t>(uv_stride) * static_cast<size_t>(height / 2);
        const size_t expected_min_size = y_size + uv_size;
        if (msg->data.size() < expected_min_size)
        {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "NV12 image data too small: got=%zu expected_at_least=%zu",
            msg->data.size(), expected_min_size);
          return;
        }

        if (!EnsureCudaStream())
        {
          return;
        }
        if (!EnsureCudaBuffer(&cuda_input_copy_buffer_, cuda_input_copy_buffer_size_, expected_min_size))
        {
          return;
        }

        cudaError_t err = cudaMemcpyAsync(
          cuda_input_copy_buffer_, msg->data.data(), expected_min_size,
          cudaMemcpyHostToDevice, cuda_stream_);
        if (err != cudaSuccess)
        {
          RCLCPP_WARN(get_logger(), "cudaMemcpyAsync NV12 HostToDevice failed: %s",
            cudaGetErrorString(err));
          return;
        }

        const int bgr_stride = width * 3;
        const size_t bgr_size =
          static_cast<size_t>(bgr_stride) * static_cast<size_t>(height);
        if (!EnsureCudaBuffer(&cuda_bgr_publish_buffer_, cuda_bgr_publish_buffer_size_, bgr_size))
        {
          return;
        }

        err = ConvertGxfNV12ToBGREx(
          cuda_bgr_publish_buffer_, cuda_input_copy_buffer_, width, height, y_stride, uv_stride,
          bgr_stride, 0, cuda_stream_);
        if (err != cudaSuccess)
        {
          RCLCPP_WARN(get_logger(), "ConvertGxfNV12ToBGREx failed: %s", cudaGetErrorString(err));
          return;
        }

        bgr = cv::Mat(height, width, CV_8UC3);
        err = cudaMemcpy2DAsync(
          bgr.data, bgr_stride, cuda_bgr_publish_buffer_, bgr_stride, bgr_stride, height,
          cudaMemcpyDeviceToHost, cuda_stream_);
        if (err != cudaSuccess)
        {
          RCLCPP_WARN(get_logger(), "cudaMemcpy2DAsync GPU BGR failed: %s",
            cudaGetErrorString(err));
          return;
        }
        err = cudaStreamSynchronize(cuda_stream_);
        if (err != cudaSuccess)
        {
          RCLCPP_WARN(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
          return;
        }
      }
      else
      {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Unsupported sensor_msgs/Image encoding: %s",
          msg->encoding.c_str());
        return;
      }
    }
    catch (const cv::Exception &ex)
    {
      RCLCPP_WARN(
        get_logger(), "OpenCV conversion failed for encoding %s: %s",
        msg->encoding.c_str(), ex.what());
      return;
    }

    std::vector<LidarPoint> cloud_points;
    rclcpp::Time cloud_stamp(0, 0, get_clock()->get_clock_type());
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      cloud_points = last_cloud_points_;
      cloud_stamp = last_cloud_stamp_;
    }

    const rclcpp::Time image_stamp(msg->header.stamp);
    const double cloud_age_ms =
      cloud_points.empty() ? std::numeric_limits<double>::infinity() :
      std::fabs((image_stamp - cloud_stamp).seconds()) * 1000.0;
    if (cloud_age_ms > static_cast<double>(cloud_frame_timeout_ms_))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Cloud is stale: %.1f ms > %d ms", cloud_age_ms, cloud_frame_timeout_ms_);
    }

    cv::Mat overlay = bgr.clone();
    DrawProjectedCloud(overlay, cloud_points, image_stamp, cloud_stamp);
    DrawOverlayLegend(overlay, cloud_points, image_stamp, cloud_stamp);

    UpdateLatestBgrFrame(bgr, image_stamp, msg->header.frame_id);

    PublishNitrosNv12Image(overlay, msg->header);

    if (draw_bev_on_image_callback_)
    {
      cv::Mat bev = BuildBevImage(cloud_points, image_stamp, cloud_stamp);
      auto bev_header = msg->header;
      bev_header.frame_id = "projection_bev";
      auto bev_msg =
        cv_bridge::CvImage(bev_header, sensor_msgs::image_encodings::BGR8, bev)
          .toImageMsg();
      bev_pub_->publish(*bev_msg);
    }
  }

  void OnNitrosImage(const nvidia::isaac_ros::nitros::NitrosImageView &view)
  {
    const int width = static_cast<int>(view.GetWidth());
    const int height = static_cast<int>(view.GetHeight());
    if (width <= 0 || height <= 0)
    {
      return;
    }

    const unsigned char *gpu_data = view.GetGpuData();
    if (!gpu_data)
    {
      return;
    }

    const size_t total_size = static_cast<size_t>(view.GetSizeInBytes());
    const int y_stride = static_cast<int>(view.GetStride(0));
    const int uv_stride = static_cast<int>(view.GetStride(1));

    if (cuda_stream_ == nullptr)
    {
      if (cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking) != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "Failed to create CUDA stream for NV12 conversion");
        return;
      }
    }

    if (!EnsureCudaBuffer(&cuda_input_copy_buffer_, cuda_input_copy_buffer_size_, total_size))
    {
      return;
    }

    cudaError_t err = cudaMemcpyAsync(
      cuda_input_copy_buffer_, gpu_data, total_size, cudaMemcpyDeviceToDevice, cuda_stream_);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(
        get_logger(), "cudaMemcpyAsync local input copy failed: %s", cudaGetErrorString(err));
      return;
    }

    std::string encoding;
    try
    {
      encoding = view.GetEncoding();
    }
    catch (const std::exception &ex)
    {
      RCLCPP_WARN(get_logger(), "Failed to query Nitros image encoding: %s", ex.what());
      return;
    }

    cv::Mat bgr;
    const bool encoding_is_bgr =
      encoding == sensor_msgs::image_encodings::BGR8;
    const bool encoding_is_rgb =
      encoding == sensor_msgs::image_encodings::RGB8;
    const bool is_nv12 =
      encoding == "nv12" ||
      encoding == "NV12" ||
      (!encoding_is_bgr && !encoding_is_rgb && uv_stride > 0);

    if (encoding_is_rgb)
    {
      cv::Mat rgb(height, width, CV_8UC3);
      err = cudaMemcpy2DAsync(
        rgb.data, width * 3, cuda_input_copy_buffer_, y_stride, width * 3, height,
        cudaMemcpyDeviceToHost, cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "cudaMemcpy2DAsync RGB failed: %s", cudaGetErrorString(err));
        return;
      }
      err = cudaStreamSynchronize(cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
        return;
      }
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    }
    else if (encoding_is_bgr)
    {
      bgr = cv::Mat(height, width, CV_8UC3);
      err = cudaMemcpy2DAsync(
        bgr.data, width * 3, cuda_input_copy_buffer_, y_stride, width * 3, height,
        cudaMemcpyDeviceToHost, cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "cudaMemcpy2DAsync BGR failed: %s", cudaGetErrorString(err));
        return;
      }
    }
    else if (is_nv12)
    {
      if (y_stride < width || uv_stride < width || (height % 2) != 0)
      {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Unsupported NV12 layout: encoding=%s width=%d height=%d y_stride=%d uv_stride=%d",
          encoding.c_str(), width, height, y_stride, uv_stride);
        return;
      }

      const int bgr_stride = width * 3;
      const size_t bgr_size =
        static_cast<size_t>(bgr_stride) * static_cast<size_t>(height);
      if (!EnsureCudaBuffer(&cuda_bgr_publish_buffer_, cuda_bgr_publish_buffer_size_, bgr_size))
      {
        return;
      }

      const bool use_nv21 =
        (input_nv12_variant_ == "nv21" || input_nv12_variant_ == "NV21" ||
         input_nv12_variant_ == "vu");
      err = ConvertGxfNV12ToBGREx(
        cuda_bgr_publish_buffer_, cuda_input_copy_buffer_, width, height, y_stride, uv_stride,
        bgr_stride, use_nv21 ? 1 : 0, cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "ConvertGxfNV12ToBGREx failed: %s", cudaGetErrorString(err));
        return;
      }

      bgr = cv::Mat(height, width, CV_8UC3);
      err = cudaMemcpy2DAsync(
        bgr.data, bgr_stride, cuda_bgr_publish_buffer_, bgr_stride, bgr_stride, height,
        cudaMemcpyDeviceToHost, cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "cudaMemcpy2DAsync GPU BGR failed: %s", cudaGetErrorString(err));
        return;
      }
      err = cudaStreamSynchronize(cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
        return;
      }
    }
    else if (y_stride >= width * 3 && uv_stride == 0)
    {
      // Fallback for packed 3-channel images when the encoding string is missing
      // or unhelpful. Assume RGB first since that matches observed camera output.
      cv::Mat rgb(height, width, CV_8UC3);
      err = cudaMemcpy2DAsync(
        rgb.data, width * 3, cuda_input_copy_buffer_, y_stride, width * 3, height,
        cudaMemcpyDeviceToHost, cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(
          get_logger(), "cudaMemcpy2DAsync packed RGB fallback failed: %s",
          cudaGetErrorString(err));
        return;
      }
      err = cudaStreamSynchronize(cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
        return;
      }
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    }
    else
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Unsupported Nitros encoding: %s width=%d height=%d y_stride=%d uv_stride=%d",
        encoding.c_str(), width, height, y_stride, uv_stride);
      return;
    }

    if (!encoding_is_rgb && !is_nv12 && !(y_stride >= width * 3 && uv_stride == 0))
    {
      err = cudaStreamSynchronize(cuda_stream_);
      if (err != cudaSuccess)
      {
        RCLCPP_WARN(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
        return;
      }
    }

    std::vector<LidarPoint> cloud_points;
    rclcpp::Time cloud_stamp(0, 0, get_clock()->get_clock_type());
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      cloud_points = last_cloud_points_;
      cloud_stamp = last_cloud_stamp_;
    }

    const rclcpp::Time image_stamp(
      static_cast<int32_t>(view.GetTimestampSeconds()),
      view.GetTimestampNanoseconds(),
      RCL_ROS_TIME);

    cv::Mat overlay = bgr.clone();
    DrawProjectedCloud(overlay, cloud_points, image_stamp, cloud_stamp);
    DrawOverlayLegend(overlay, cloud_points, image_stamp, cloud_stamp);

    UpdateLatestBgrFrame(bgr, image_stamp, "projection_camera");

    std_msgs::msg::Header header;
    header.stamp = image_stamp;
    header.frame_id = "projection_overlay";
    PublishNitrosNv12Image(overlay, header);

    if (draw_bev_on_image_callback_)
    {
      cv::Mat bev = BuildBevImage(cloud_points, image_stamp, cloud_stamp);
      header.frame_id = "projection_bev";
      auto bev_msg =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, bev)
          .toImageMsg();
      bev_pub_->publish(*bev_msg);
    }
  }

  bool EnsureCudaBuffer(void **buffer, size_t &buffer_size, size_t required_size)
  {
    if (*buffer != nullptr && buffer_size >= required_size)
    {
      return true;
    }
    if (*buffer != nullptr)
    {
      cudaFree(*buffer);
      *buffer = nullptr;
      buffer_size = 0;
    }
    if (cudaMalloc(buffer, required_size) != cudaSuccess)
    {
      RCLCPP_WARN(get_logger(), "Failed to allocate CUDA buffer, bytes=%zu", required_size);
      return false;
    }
    buffer_size = required_size;
    return true;
  }

  bool EnsureCudaStream()
  {
    if (cuda_stream_ != nullptr)
    {
      return true;
    }
    const cudaError_t err = cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(get_logger(), "Failed to create CUDA stream: %s", cudaGetErrorString(err));
      return false;
    }
    return true;
  }

  void UpdateLatestBgrFrame(
    const cv::Mat &bgr, const rclcpp::Time &stamp, const std::string &frame_id)
  {
    if (!publish_colored_cloud_ || bgr.empty())
    {
      return;
    }
    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_bgr_frame_ = bgr.clone();
    latest_image_stamp_ = stamp;
    latest_image_frame_id_ = frame_id;
  }

  void PublishColoredCloudFromLatestImage(const rclcpp::Time &cloud_stamp)
  {
    std::vector<LidarPoint> cloud_points;
    cv::Mat image;
    rclcpp::Time image_stamp(0, 0, get_clock()->get_clock_type());
    std::string cloud_frame_id;

    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      cloud_points = last_cloud_points_;
      cloud_frame_id = last_cloud_frame_id_;
    }
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if (!latest_bgr_frame_.empty())
      {
        image = latest_bgr_frame_.clone();
        image_stamp = latest_image_stamp_;
      }
    }

    if (cloud_points.empty())
    {
      return;
    }

    cv::Mat bev = BuildBevImage(cloud_points, image_stamp, cloud_stamp);
    std_msgs::msg::Header bev_header;
    bev_header.stamp = cloud_stamp;
    bev_header.frame_id = "projection_bev";
    auto bev_msg =
      cv_bridge::CvImage(bev_header, sensor_msgs::image_encodings::BGR8, bev).toImageMsg();
    bev_pub_->publish(*bev_msg);

    if (!publish_colored_cloud_ || !colored_cloud_pub_ || image.empty())
    {
      return;
    }

    const double image_age_ms =
      std::fabs((cloud_stamp - image_stamp).seconds()) * 1000.0;
    if (image_age_ms > static_cast<double>(camera_frame_timeout_ms_))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Image is stale for colored cloud: %.1f ms > %d ms",
        image_age_ms, camera_frame_timeout_ms_);
      return;
    }

    std::vector<ColoredPoint> frame_colored_points;
    frame_colored_points.reserve(cloud_points.size());
    for (const auto &point : cloud_points)
    {
      if (!IsPointColorable(point, image.cols, image.rows))
      {
        continue;
      }
      const int u = static_cast<int>(std::lround(point.uv.x));
      const int v = static_cast<int>(std::lround(point.uv.y));
      const cv::Vec3b bgr = image.at<cv::Vec3b>(v, u);
      const uint32_t rgb =
        (static_cast<uint32_t>(bgr[2]) << 16) |
        (static_cast<uint32_t>(bgr[1]) << 8) |
        static_cast<uint32_t>(bgr[0]);
      frame_colored_points.push_back(
        ColoredPoint{point.lidar[0], point.lidar[1], point.lidar[2], rgb});
    }

    if (frame_colored_points.empty())
    {
      return;
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = cloud_stamp;
    cloud_msg.header.frame_id = cloud_frame_id.empty() ? "livox_frame" : cloud_frame_id;
    cloud_msg.height = 1;
    cloud_msg.is_bigendian = false;
    cloud_msg.is_dense = false;

    if (!enable_colored_cloud_sliding_window_)
    {
      {
        std::lock_guard<std::mutex> lock(colored_cloud_window_mutex_);
        colored_cloud_window_.clear();
        colored_cloud_window_point_count_ = 0;
      }

      sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
      modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
      modifier.resize(frame_colored_points.size());
      cloud_msg.width = static_cast<uint32_t>(frame_colored_points.size());

      sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
      sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud_msg, "rgb");
      for (const auto &point : frame_colored_points)
      {
        float rgb_float = 0.0f;
        std::memcpy(&rgb_float, &point.rgb, sizeof(rgb_float));

        *iter_x = point.x;
        *iter_y = point.y;
        *iter_z = point.z;
        *iter_rgb = rgb_float;
        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_rgb;
      }

      colored_cloud_pub_->publish(cloud_msg);
      return;
    }

    {
      std::lock_guard<std::mutex> lock(colored_cloud_window_mutex_);
      colored_cloud_window_.push_back(
        ColoredCloudBatch{cloud_stamp, std::move(frame_colored_points)});
      colored_cloud_window_point_count_ += colored_cloud_window_.back().points.size();
      PruneColoredCloudWindowLocked(cloud_stamp);
      if (colored_cloud_window_point_count_ == 0)
      {
        return;
      }

      sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
      modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
      modifier.resize(colored_cloud_window_point_count_);
      cloud_msg.width = static_cast<uint32_t>(colored_cloud_window_point_count_);

      sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
      sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud_msg, "rgb");
      for (const auto &batch : colored_cloud_window_)
      {
        for (const auto &point : batch.points)
        {
          float rgb_float = 0.0f;
          std::memcpy(&rgb_float, &point.rgb, sizeof(rgb_float));

          *iter_x = point.x;
          *iter_y = point.y;
          *iter_z = point.z;
          *iter_rgb = rgb_float;
          ++iter_x;
          ++iter_y;
          ++iter_z;
          ++iter_rgb;
        }
      }
    }

    colored_cloud_pub_->publish(cloud_msg);
  }

  void PruneColoredCloudWindowLocked(const rclcpp::Time &now)
  {
    const auto max_age = rclcpp::Duration(
      colored_cloud_window_ms_ / 1000,
      static_cast<uint32_t>((colored_cloud_window_ms_ % 1000) * 1000000));
    while (!colored_cloud_window_.empty() &&
           (now - colored_cloud_window_.front().stamp) > max_age)
    {
      const size_t front_size = colored_cloud_window_.front().points.size();
      colored_cloud_window_point_count_ =
        (colored_cloud_window_point_count_ > front_size) ?
        (colored_cloud_window_point_count_ - front_size) : 0;
      colored_cloud_window_.pop_front();
    }
    while (colored_cloud_window_point_count_ > static_cast<size_t>(colored_cloud_max_points_) &&
           !colored_cloud_window_.empty())
    {
      const size_t front_size = colored_cloud_window_.front().points.size();
      colored_cloud_window_point_count_ =
        (colored_cloud_window_point_count_ > front_size) ?
        (colored_cloud_window_point_count_ - front_size) : 0;
      colored_cloud_window_.pop_front();
    }
  }

  bool IsPointColorable(const LidarPoint &point, int image_width, int image_height) const
  {
    if (!point.projected)
    {
      return false;
    }
    if (overlay_only_front_points_ && point.lidar[0] <= 0.0f)
    {
      return false;
    }
    if (!(point.camera[2] > 1e-6f))
    {
      return false;
    }
    const int u = static_cast<int>(std::lround(point.uv.x));
    const int v = static_cast<int>(std::lround(point.uv.y));
    return u >= 0 && u < image_width && v >= 0 && v < image_height;
  }

  void PublishNitrosNv12Image(const cv::Mat &bgr, const std_msgs::msg::Header &header)
  {
    if (!nitros_overlay_pub_ || bgr.empty())
    {
      return;
    }

    const int width = bgr.cols;
    const int height = bgr.rows;
    if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Projection overlay has invalid NV12 dimensions: width=%d height=%d",
        width, height);
      return;
    }

    if (!EnsureCudaStream())
    {
      return;
    }

    const int bgr_stride = width * 3;
    const int y_stride = width;
    const int uv_stride = width * 2;
    const size_t bgr_size =
      static_cast<size_t>(bgr_stride) * static_cast<size_t>(height);
    const size_t nv12_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 2U;

    if (!EnsureCudaBuffer(&cuda_bgr_publish_buffer_, cuda_bgr_publish_buffer_size_, bgr_size))
    {
      return;
    }

    if (!nitros_publish_pool_ || nitros_publish_pool_->size_bytes != nv12_size)
    {
      nitros_publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
      nitros_publish_pool_->warmup(2);
    }

    void *publish_buffer = nitros_publish_pool_ ? nitros_publish_pool_->acquire() : nullptr;
    if (!publish_buffer)
    {
      RCLCPP_WARN(get_logger(), "Failed to acquire NV12 publish buffer");
      return;
    }

    cudaError_t err = cudaMemcpyAsync(
      cuda_bgr_publish_buffer_, bgr.data, bgr_size, cudaMemcpyHostToDevice, cuda_stream_);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(
        get_logger(), "cudaMemcpyAsync HostToDevice failed: %s", cudaGetErrorString(err));
      nitros_publish_pool_->release(publish_buffer);
      return;
    }

    err = ConvertBGRToGxfNV12(
      publish_buffer, cuda_bgr_publish_buffer_, width, height, y_stride, uv_stride, bgr_stride,
      cuda_stream_);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(get_logger(), "ConvertBGRToGxfNV12 failed: %s", cudaGetErrorString(err));
      nitros_publish_pool_->release(publish_buffer);
      return;
    }

    err = cudaStreamSynchronize(cuda_stream_);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
      nitros_publish_pool_->release(publish_buffer);
      return;
    }

    try
    {
      auto nitros_image = BuildNv12NitrosImageWithPool(
        header, static_cast<uint32_t>(height), static_cast<uint32_t>(width),
        publish_buffer, y_stride, uv_stride, nitros_publish_pool_);
      nitros_overlay_pub_->publish(nitros_image);
      // RCLCPP_INFO_THROTTLE(
      //   get_logger(), *get_clock(), 5000,
      //   "Published projection NV12: topic=%s width=%d height=%d y_stride=%d uv_stride=%d",
      //   overlay_topic_.c_str(), width, height, y_stride, uv_stride);
    }
    catch (const std::exception &ex)
    {
      RCLCPP_WARN(get_logger(), "Failed to publish Nitros NV12 image: %s", ex.what());
      nitros_publish_pool_->release(publish_buffer);
    }
  }

  void DrawProjectedCloud(
    cv::Mat &overlay, const std::vector<LidarPoint> &cloud_points,
    const rclcpp::Time &image_stamp, const rclcpp::Time &cloud_stamp)
  {
    (void)image_stamp;
    (void)cloud_stamp;
    if (overlay.empty())
    {
      return;
    }

    int index = 0;
    for (const auto &point : cloud_points)
    {
      if ((index++ % draw_every_nth_point_) != 0)
      {
        continue;
      }
      if (overlay_only_front_points_ && point.lidar[0] <= 0.0f)
      {
        continue;
      }
      if (!point.projected)
      {
        continue;
      }
      if (point.uv.x < 0.0f || point.uv.x >= static_cast<float>(overlay.cols) ||
          point.uv.y < 0.0f || point.uv.y >= static_cast<float>(overlay.rows))
      {
        continue;
      }

      const float depth = point.lidar[0];
      const float lateral = point.lidar[1];
      const float normalized =
        static_cast<float>(Clamp((depth - 0.5) / std::max(0.5, bev_x_max_m_ - bev_x_min_m_), 0.0, 1.0));
      const int red = static_cast<int>(255.0f * (1.0f - normalized));
      const int green = static_cast<int>(255.0f * normalized);
      const int blue = static_cast<int>(Clamp((lateral - bev_y_min_m_) / std::max(0.1, bev_y_max_m_ - bev_y_min_m_), 0.0, 1.0) * 255.0);
      cv::circle(
        overlay, cv::Point(static_cast<int>(std::lround(point.uv.x)), static_cast<int>(std::lround(point.uv.y))),
        point_radius_px_, cv::Scalar(blue, green, red), cv::FILLED, cv::LINE_AA);
    }
  }

  void DrawOverlayLegend(
    cv::Mat &overlay, const std::vector<LidarPoint> &cloud_points,
    const rclcpp::Time &image_stamp, const rclcpp::Time &cloud_stamp)
  {
    if (overlay.empty())
    {
      return;
    }

    const double cloud_age_ms =
      cloud_points.empty() ? std::numeric_limits<double>::infinity() :
      std::fabs((image_stamp - cloud_stamp).seconds()) * 1000.0;

    cv::rectangle(overlay, cv::Rect(16, 16, 520, 110), cv::Scalar(24, 24, 24), cv::FILLED);
    cv::putText(
      overlay, "Camera-Lidar Projection Demo", cv::Point(28, 46),
      cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(240, 240, 240), 2, cv::LINE_AA);

    std::ostringstream info1;
    info1 << "Points: " << cloud_points.size()
          << "  Cloud age: ";
    if (std::isfinite(cloud_age_ms))
    {
      info1 << std::fixed << std::setprecision(1) << cloud_age_ms << " ms";
    }
    else
    {
      info1 << "n/a";
    }
    cv::putText(
      overlay, info1.str(), cv::Point(28, 76),
      cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(210, 210, 210), 2, cv::LINE_AA);

    cv::putText(
      overlay, "Color: red=near, green=far, blue=lateral", cv::Point(28, 104),
      cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(210, 210, 210), 2, cv::LINE_AA);
  }

  cv::Mat BuildBevImage(
    const std::vector<LidarPoint> &cloud_points, const rclcpp::Time &image_stamp,
    const rclcpp::Time &cloud_stamp)
  {
    (void)image_stamp;
    (void)cloud_stamp;
    cv::Mat bev(
      std::max(64, bev_height_), std::max(64, bev_width_), CV_8UC3,
      cv::Scalar(18, 22, 26));

    DrawBevGrid(bev);

    for (size_t i = 0; i < cloud_points.size(); ++i)
    {
      if ((static_cast<int>(i) % draw_every_nth_point_) != 0)
      {
        continue;
      }
      const auto &p = cloud_points[i].lidar;
      const double x = static_cast<double>(p[0]);
      const double y = static_cast<double>(p[1]);
      const double z = static_cast<double>(p[2]);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      {
        continue;
      }
      if (x < bev_x_min_m_ || x > bev_x_max_m_ || y < bev_y_min_m_ || y > bev_y_max_m_)
      {
        continue;
      }

      const double y_span = std::max(0.1, bev_y_max_m_ - bev_y_min_m_);
      const double u_ratio =
        bev_flip_y_ ? (bev_y_max_m_ - y) / y_span : (y - bev_y_min_m_) / y_span;
      const double v_ratio = (bev_x_max_m_ - x) / std::max(0.1, bev_x_max_m_ - bev_x_min_m_);
      const int u = static_cast<int>(std::lround(u_ratio * static_cast<double>(bev.cols - 1)));
      const int v = static_cast<int>(std::lround(v_ratio * static_cast<double>(bev.rows - 1)));
      const int height_color = static_cast<int>(
        Clamp((z + 2.0) / 4.0, 0.0, 1.0) * 255.0);
      cv::circle(
        bev, cv::Point(u, v), 1,
        cv::Scalar(255 - height_color, height_color, 180), cv::FILLED, cv::LINE_AA);
    }

    const int ego_x = static_cast<int>(std::lround(
      (bev_flip_y_ ?
      ((bev_y_max_m_ - 0.0) / std::max(0.1, bev_y_max_m_ - bev_y_min_m_)) :
      ((0.0 - bev_y_min_m_) / std::max(0.1, bev_y_max_m_ - bev_y_min_m_))) *
      static_cast<double>(bev.cols - 1)));
    const int ego_y = bev.rows - 24;
    cv::drawMarker(
      bev, cv::Point(ego_x, ego_y), cv::Scalar(0, 255, 255), cv::MARKER_TRIANGLE_UP,
      28, 2, cv::LINE_AA);
    cv::putText(
      bev, "EGO", cv::Point(std::max(0, ego_x - 18), std::max(24, ego_y - 18)),
      cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    cv::putText(
      bev, "BEV (lidar frame)", cv::Point(20, 34), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(240, 240, 240), 2, cv::LINE_AA);

    return bev;
  }

  void DrawBevGrid(cv::Mat &bev)
  {
    const double x_span = std::max(0.1, bev_x_max_m_ - bev_x_min_m_);
    const double y_span = std::max(0.1, bev_y_max_m_ - bev_y_min_m_);

    for (double x = std::ceil(bev_x_min_m_ / grid_step_m_) * grid_step_m_;
         x <= bev_x_max_m_ + 1e-6; x += grid_step_m_)
    {
      const double v_ratio = (bev_x_max_m_ - x) / x_span;
      const int v = static_cast<int>(std::lround(v_ratio * static_cast<double>(bev.rows - 1)));
      cv::line(bev, cv::Point(0, v), cv::Point(bev.cols - 1, v), cv::Scalar(50, 58, 66), 1, cv::LINE_AA);
      std::ostringstream label;
      label << std::fixed << std::setprecision(0) << x << "m";
      cv::putText(
        bev, label.str(), cv::Point(8, std::max(16, v - 4)),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(140, 150, 160), 1, cv::LINE_AA);
    }

    for (double y = std::ceil(bev_y_min_m_ / grid_step_m_) * grid_step_m_;
         y <= bev_y_max_m_ + 1e-6; y += grid_step_m_)
    {
      const double u_ratio =
        bev_flip_y_ ? (bev_y_max_m_ - y) / y_span : (y - bev_y_min_m_) / y_span;
      const int u = static_cast<int>(std::lround(u_ratio * static_cast<double>(bev.cols - 1)));
      cv::line(bev, cv::Point(u, 0), cv::Point(u, bev.rows - 1), cv::Scalar(50, 58, 66), 1, cv::LINE_AA);
    }

    const int center_u = static_cast<int>(std::lround(
      (bev_flip_y_ ? ((bev_y_max_m_ - 0.0) / y_span) : ((0.0 - bev_y_min_m_) / y_span)) *
      static_cast<double>(bev.cols - 1)));
    cv::line(bev, cv::Point(center_u, 0), cv::Point(center_u, bev.rows - 1), cv::Scalar(80, 160, 255), 2, cv::LINE_AA);
  }

  std::string image_topic_;
  bool use_nitros_image_{true};
  std::string nitros_image_format_{"nitros_image_nv12"};
  std::string lidar_topic_;
  bool use_livox_custom_msg_{true};
  std::string overlay_topic_;
  std::string bev_topic_;
  std::string colored_cloud_topic_;
  std::string calib_result_path_;

  int bev_width_{800};
  int bev_height_{1200};
  double bev_x_min_m_{0.0};
  double bev_x_max_m_{15.0};
  double bev_y_min_m_{-5.0};
  double bev_y_max_m_{5.0};
  bool bev_flip_y_{true};
  int draw_every_nth_point_{2};
  int point_radius_px_{2};
  int camera_frame_timeout_ms_{300};
  int cloud_frame_timeout_ms_{300};
  double grid_step_m_{1.0};
  bool overlay_only_front_points_{true};
  bool publish_colored_cloud_{true};
  bool enable_colored_cloud_sliding_window_{true};
  int colored_cloud_window_ms_{1000};
  int colored_cloud_max_points_{200000};
  bool enable_cloud_filter_{true};
  double filter_min_range_m_{0.5};
  double filter_max_range_m_{80.0};
  double filter_min_z_m_{-2.5};
  double filter_max_z_m_{3.0};
  double filter_voxel_leaf_size_m_{0.08};
  double filter_radius_m_{0.25};
  int filter_min_neighbors_{2};
  bool draw_bev_on_image_callback_{false};
  std::string input_nv12_variant_{"nv21"};

  CameraModel camera_model_;

  std::mutex cloud_mutex_;
  std::vector<LidarPoint> last_cloud_points_;
  rclcpp::Time last_cloud_stamp_{0, 0, RCL_ROS_TIME};
  std::string last_cloud_frame_id_{"livox_frame"};

  std::mutex image_mutex_;
  cv::Mat latest_bgr_frame_;
  rclcpp::Time latest_image_stamp_{0, 0, RCL_ROS_TIME};
  std::string latest_image_frame_id_;

  std::mutex colored_cloud_window_mutex_;
  std::deque<ColoredCloudBatch> colored_cloud_window_;
  size_t colored_cloud_window_point_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
    nvidia::isaac_ros::nitros::NitrosImageView>> nitros_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_sub_;
#if ISAAC_ROS_CAM_AND_LIDAR_HAS_LIVOX_CUSTOM_MSG
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_;
#endif
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
    nvidia::isaac_ros::nitros::NitrosImage>> nitros_overlay_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bev_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_cloud_pub_;
  rclcpp::CallbackGroup::SharedPtr image_callback_group_;
  rclcpp::CallbackGroup::SharedPtr lidar_callback_group_;
  void *cuda_input_copy_buffer_{nullptr};
  size_t cuda_input_copy_buffer_size_{0};
  void *cuda_bgr_publish_buffer_{nullptr};
  size_t cuda_bgr_publish_buffer_size_{0};
  cudaStream_t cuda_stream_{nullptr};
  std::shared_ptr<Nv12PublishPool> nitros_publish_pool_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraLidarProjectionDemo>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(CameraLidarProjectionDemo)
