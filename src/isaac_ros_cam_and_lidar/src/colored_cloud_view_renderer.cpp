#include <cuda_runtime.h>
#include <isaac_ros_managed_nitros/managed_nitros_publisher.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
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
      std::lock_guard<std::mutex> lock(m);
      for (void *ptr : free_list)
      {
        if (ptr)
        {
          cudaFree(ptr);
        }
      }
    }

    void *acquire()
    {
      std::lock_guard<std::mutex> lock(m);
      if (!free_list.empty())
      {
        void *ptr = free_list.back();
        free_list.pop_back();
        return ptr;
      }

      void *ptr = nullptr;
      if (cudaMalloc(&ptr, size_bytes) != cudaSuccess)
      {
        return nullptr;
      }
      return ptr;
    }

    void release(void *ptr)
    {
      if (!ptr)
      {
        return;
      }
      std::lock_guard<std::mutex> lock(m);
      free_list.push_back(ptr);
    }

    void warmup(size_t count)
    {
      std::lock_guard<std::mutex> lock(m);
      while (free_list.size() < count)
      {
        void *ptr = nullptr;
        if (cudaMalloc(&ptr, size_bytes) != cudaSuccess)
        {
          break;
        }
        free_list.push_back(ptr);
      }
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
      throw std::runtime_error(error_msg.str());
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
      throw std::runtime_error(error_msg.str());
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
      throw std::runtime_error(error_msg.str());
    }

    using GxfVideoFormat = nvidia::gxf::VideoFormat;
    nvidia::gxf::VideoFormatSize<GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER> format_size;
    std::array<nvidia::gxf::ColorPlane, 2> planes{
        nvidia::gxf::ColorPlane("Y", 1, static_cast<uint32_t>(y_stride)),
        nvidia::gxf::ColorPlane("UV", 2, static_cast<uint32_t>(uv_stride)),
    };
    const uint64_t size = format_size.size(width, height, planes, false);
    std::vector<nvidia::gxf::ColorPlane> color_planes{planes.begin(), planes.end()};

    nvidia::gxf::VideoBufferInfo buffer_info{
        width,
        height,
        GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER,
        std::move(color_planes),
        nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};

    gxf_image.value()->wrapMemory(
        buffer_info, size, nvidia::gxf::MemoryStorageType::kDevice, gpu_data,
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

class ColoredCloudViewRenderer : public rclcpp::Node
{
public:
  explicit ColoredCloudViewRenderer(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : Node("colored_cloud_view_renderer", options)
  {
    const int width_param = static_cast<int>(declare_parameter<int64_t>("width", 1280));
    const int height_param = static_cast<int>(declare_parameter<int64_t>("height", 720));
    const int max_points_param = static_cast<int>(
        declare_parameter<int64_t>("max_points", 200000));
    const int background_b_param = static_cast<int>(
        declare_parameter<int64_t>("background_b", 16));
    const int background_g_param = static_cast<int>(
        declare_parameter<int64_t>("background_g", 20));
    const int background_r_param = static_cast<int>(
        declare_parameter<int64_t>("background_r", 24));
    const int stale_cloud_timeout_ms_param = static_cast<int>(
        declare_parameter<int64_t>("stale_cloud_timeout_ms", 2000));
    const bool publish_enabled = declare_parameter<bool>("publish_enabled", true);

    cloud_topic_ = declare_parameter<std::string>(
        "cloud_topic", "/front/projection_colored_cloud");
    output_topic_ = declare_parameter<std::string>(
        "output_topic", "/front/colored_cloud_view");
    frame_id_ = declare_parameter<std::string>("frame_id", "colored_cloud_view");
    width_ = MakeEven(std::max(64, width_param));
    height_ = MakeEven(std::max(64, height_param));
    fps_ = std::max(1.0, declare_parameter<double>("fps", 20.0));
    yaw_deg_ = declare_parameter<double>("yaw_deg", 90.0);
    pitch_deg_ = declare_parameter<double>("pitch_deg", -8.0);
    roll_deg_ = declare_parameter<double>("roll_deg", 0.0);
    camera_distance_m_ = std::max(0.1, declare_parameter<double>("camera_distance_m", 16.0));
    look_at_x_m_ = declare_parameter<double>("look_at_x_m", 8.0);
    look_at_y_m_ = declare_parameter<double>("look_at_y_m", 0.0);
    look_at_z_m_ = declare_parameter<double>("look_at_z_m", 0.3);
    focal_length_px_ = std::max(10.0, declare_parameter<double>("focal_length_px", 850.0));
    max_points_ = std::max(1, max_points_param);
    splat_radius_min_px_ =
        std::max(1, static_cast<int>(declare_parameter<int64_t>("splat_radius_min_px", 2)));
    splat_radius_max_px_ =
        std::max(splat_radius_min_px_,
                 static_cast<int>(declare_parameter<int64_t>("splat_radius_max_px", 7)));
    splat_depth_falloff_m_ =
        std::max(1.0, declare_parameter<double>("splat_depth_falloff_m", 18.0));
    splat_outer_alpha_ = Clamp(declare_parameter<double>("splat_outer_alpha", 0.32), 0.05, 1.0);
    splat_inner_alpha_ = Clamp(declare_parameter<double>("splat_inner_alpha", 0.85), 0.05, 1.0);
    splat_ellipse_ratio_ =
        Clamp(declare_parameter<double>("splat_ellipse_ratio", 1.35), 1.0, 3.0);
    splat_draw_core_ = declare_parameter<bool>("splat_draw_core", true);
    draw_grid_ = declare_parameter<bool>("draw_grid", true);
    grid_step_m_ = std::max(0.5, declare_parameter<double>("grid_step_m", 1.0));
    grid_extent_m_ = std::max(1.0, declare_parameter<double>("grid_extent_m", 20.0));
    stale_cloud_timeout_ms_ = std::max(0, stale_cloud_timeout_ms_param);
    publish_enabled_ = publish_enabled;
    background_b_ = ClampColor(background_b_param);
    background_g_ = ClampColor(background_g_param);
    background_r_ = ClampColor(background_r_param);

    RecomputeView();

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ColoredCloudViewRenderer::OnCloud, this, std::placeholders::_1));

    image_pub_ = std::make_shared<
        nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
            nvidia::isaac_ros::nitros::NitrosImage>>(
        this, output_topic_,
        nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name);

    const auto period = std::chrono::duration<double>(1.0 / fps_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&ColoredCloudViewRenderer::OnTimer, this));

    RCLCPP_INFO(
        get_logger(),
        "Rendering colored cloud view: cloud=%s output=%s size=%dx%d yaw=%.1f pitch=%.1f",
        cloud_topic_.c_str(), output_topic_.c_str(), width_, height_, yaw_deg_, pitch_deg_);
  }

  ~ColoredCloudViewRenderer() override
  {
    if (cuda_bgr_buffer_ != nullptr)
    {
      cudaFree(cuda_bgr_buffer_);
      cuda_bgr_buffer_ = nullptr;
      cuda_bgr_buffer_size_ = 0;
    }
    if (cuda_stream_ != nullptr)
    {
      cudaStreamDestroy(cuda_stream_);
      cuda_stream_ = nullptr;
    }
  }

private:
  struct RenderPoint
  {
    cv::Vec3f p;
    cv::Vec3b bgr;
  };

  struct ProjectedPoint
  {
    cv::Point pixel;
    cv::Vec3b bgr;
    float depth{0.0f};
  };

  static int MakeEven(int value)
  {
    return (value % 2 == 0) ? value : value + 1;
  }

  static int ClampColor(int value)
  {
    return std::max(0, std::min(255, value));
  }

  static cv::Vec3b UnpackRgb(float rgb_float)
  {
    uint32_t packed = 0;
    std::memcpy(&packed, &rgb_float, sizeof(packed));
    const uint8_t r = static_cast<uint8_t>((packed >> 16) & 0xFF);
    const uint8_t g = static_cast<uint8_t>((packed >> 8) & 0xFF);
    const uint8_t b = static_cast<uint8_t>(packed & 0xFF);
    return cv::Vec3b(b, g, r);
  }

  cv::Vec3f ComputeForwardAxis() const
  {
    const double yaw = yaw_deg_ * CV_PI / 180.0;
    const double pitch = pitch_deg_ * CV_PI / 180.0;
    cv::Vec3f forward(
        static_cast<float>(std::cos(pitch) * std::cos(yaw)),
        static_cast<float>(std::cos(pitch) * std::sin(yaw)),
        static_cast<float>(std::sin(pitch)));
    return forward / std::max(1e-5f, static_cast<float>(cv::norm(forward)));
  }

  void ApplyViewAxes(const cv::Vec3f &forward) const
  {
    const double roll = roll_deg_ * CV_PI / 180.0;

    const cv::Vec3f world_up(0.0f, 0.0f, 1.0f);
    right_axis_ = forward.cross(world_up);
    if (cv::norm(right_axis_) < 1e-5f)
    {
      right_axis_ = cv::Vec3f(1.0f, 0.0f, 0.0f);
    }
    right_axis_ /= cv::norm(right_axis_);
    up_axis_ = right_axis_.cross(forward);
    up_axis_ /= std::max(1e-5f, static_cast<float>(cv::norm(up_axis_)));
    forward_axis_ = forward / std::max(1e-5f, static_cast<float>(cv::norm(forward)));

    if (std::fabs(roll) > 1e-6)
    {
      const cv::Vec3f old_right = right_axis_;
      const cv::Vec3f old_up = up_axis_;
      right_axis_ =
          old_right * static_cast<float>(std::cos(roll)) +
          old_up * static_cast<float>(std::sin(roll));
      up_axis_ =
          old_up * static_cast<float>(std::cos(roll)) -
          old_right * static_cast<float>(std::sin(roll));
    }
  }

  void RecomputeView()
  {
    const cv::Vec3f look_at(
        static_cast<float>(look_at_x_m_),
        static_cast<float>(look_at_y_m_),
        static_cast<float>(look_at_z_m_));
    const cv::Vec3f forward = ComputeForwardAxis();
    camera_position_ = look_at - forward * static_cast<float>(camera_distance_m_);
    ApplyViewAxes(forward);
  }

  void OnCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    std::vector<RenderPoint> points;
    points.reserve(std::min<size_t>(
        static_cast<size_t>(max_points_),
        static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height)));

    try
    {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
      sensor_msgs::PointCloud2ConstIterator<float> iter_rgb(*msg, "rgb");
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_rgb)
      {
        if (points.size() >= static_cast<size_t>(max_points_))
        {
          break;
        }
        const float x = *iter_x;
        const float y = *iter_y;
        const float z = *iter_z;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
          continue;
        }
        points.push_back(RenderPoint{cv::Vec3f(x, y, z), UnpackRgb(*iter_rgb)});
      }
    }
    catch (const std::exception &ex)
    {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Failed to parse colored PointCloud2: %s", ex.what());
      return;
    }

    std::lock_guard<std::mutex> lock(points_mutex_);
    latest_points_ = std::move(points);
    latest_cloud_stamp_ = rclcpp::Time(msg->header.stamp);
    latest_cloud_frame_id_ = msg->header.frame_id;
    have_cloud_ = true;
  }

  bool ProjectPoint(const cv::Vec3f &point, cv::Point &pixel, float &depth) const
  {
    const cv::Vec3f rel = point - camera_position_;
    const float x_cam = rel.dot(right_axis_);
    const float y_cam = rel.dot(up_axis_);
    const float z_cam = rel.dot(forward_axis_);
    if (!(z_cam > 0.05f))
    {
      return false;
    }

    const float u = static_cast<float>(width_) * 0.5f + static_cast<float>(focal_length_px_) * x_cam / z_cam;
    const float v = static_cast<float>(height_) * 0.5f - static_cast<float>(focal_length_px_) * y_cam / z_cam;
    if (u < 0.0f || u >= static_cast<float>(width_) || v < 0.0f || v >= static_cast<float>(height_))
    {
      return false;
    }
    pixel = cv::Point(static_cast<int>(std::lround(u)), static_cast<int>(std::lround(v)));
    depth = z_cam;
    return true;
  }

  void DrawGrid(cv::Mat &image) const
  {
    const cv::Scalar grid_color(58, 66, 74);
    const cv::Scalar center_x_color(80, 200, 255);
    const cv::Scalar center_y_color(120, 255, 120);
    const cv::Scalar center_cross_color(180, 240, 255);

    for (double x = 0.0; x <= grid_extent_m_ + 1e-6; x += grid_step_m_)
    {
      const bool is_center = std::fabs(x) <= 1e-6;
      DrawWorldLine(
          image,
          cv::Vec3f(static_cast<float>(x), static_cast<float>(-grid_extent_m_), 0.0f),
          cv::Vec3f(static_cast<float>(x), static_cast<float>(grid_extent_m_), 0.0f),
          is_center ? center_x_color : grid_color,
          is_center ? 2 : 1);
    }

    for (double y = -grid_extent_m_; y <= grid_extent_m_ + 1e-6; y += grid_step_m_)
    {
      const bool is_center = std::fabs(y) <= 1e-6;
      DrawWorldLine(
          image,
          cv::Vec3f(0.0f, static_cast<float>(y), 0.0f),
          cv::Vec3f(static_cast<float>(grid_extent_m_), static_cast<float>(y), 0.0f),
          is_center ? center_y_color : grid_color,
          is_center ? 2 : 1);
    }

    DrawWorldLine(
        image,
        cv::Vec3f(0.0f, 0.0f, 0.0f),
        cv::Vec3f(static_cast<float>(grid_extent_m_), 0.0f, 0.0f),
        center_x_color, 2);
    DrawWorldLine(
        image,
        cv::Vec3f(0.0f, static_cast<float>(-grid_extent_m_), 0.0f),
        cv::Vec3f(0.0f, static_cast<float>(grid_extent_m_), 0.0f),
        center_y_color, 2);

    cv::Point center_pixel;
    float center_depth = 0.0f;
    if (ProjectPoint(cv::Vec3f(0.0f, 0.0f, 0.0f), center_pixel, center_depth))
    {
      cv::drawMarker(
          image, center_pixel, center_cross_color, cv::MARKER_CROSS,
          16, 2, cv::LINE_AA);
    }
  }

  void DrawWorldLine(
      cv::Mat &image, const cv::Vec3f &a, const cv::Vec3f &b,
      const cv::Scalar &color, int thickness = 1) const
  {
    cv::Point pa;
    cv::Point pb;
    float da = 0.0f;
    float db = 0.0f;
    if (ProjectPoint(a, pa, da) && ProjectPoint(b, pb, db))
    {
      cv::line(image, pa, pb, color, thickness, cv::LINE_AA);
    }
  }

  std::vector<ProjectedPoint> BuildProjectedPoints(const std::vector<RenderPoint> &points) const
  {
    std::vector<ProjectedPoint> projected;
    projected.reserve(points.size());
    for (const auto &point : points)
    {
      cv::Point pixel;
      float depth = 0.0f;
      if (ProjectPoint(point.p, pixel, depth))
      {
        projected.push_back(ProjectedPoint{pixel, point.bgr, depth});
      }
    }
    return projected;
  }

  int ComputeSplatRadiusPx(float depth) const
  {
    const double t = Clamp(
        1.0 - static_cast<double>(depth) / splat_depth_falloff_m_, 0.0, 1.0);
    return splat_radius_min_px_ + static_cast<int>(std::lround(
               t * static_cast<double>(splat_radius_max_px_ - splat_radius_min_px_)));
  }

  void BlendFilledEllipse(
      cv::Mat &image, const cv::Point &center, const cv::Size &axes,
      const cv::Scalar &color, double alpha) const
  {
    const cv::Rect full_bounds(0, 0, image.cols, image.rows);
    cv::Rect bounds(
        center.x - axes.width - 2, center.y - axes.height - 2,
        axes.width * 2 + 4, axes.height * 2 + 4);
    bounds &= full_bounds;
    if (bounds.width <= 0 || bounds.height <= 0)
    {
      return;
    }

    cv::Mat roi = image(bounds);
    cv::Mat overlay = roi.clone();
    const cv::Point shifted_center(center.x - bounds.x, center.y - bounds.y);
    cv::ellipse(
        overlay, shifted_center, axes, 0.0, 0.0, 360.0, color,
        cv::FILLED, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
  }

  void DrawSplats(cv::Mat &image, std::vector<ProjectedPoint> projected) const
  {
    std::sort(
        projected.begin(), projected.end(),
        [](const ProjectedPoint &lhs, const ProjectedPoint &rhs)
        {
          return lhs.depth > rhs.depth;
        });

    for (const auto &point : projected)
    {
      const double fade = Clamp(1.0 - static_cast<double>(point.depth) / 80.0, 0.30, 1.0);
      const cv::Scalar base_color(
          point.bgr[0] * fade,
          point.bgr[1] * fade,
          point.bgr[2] * fade);
      const int radius = ComputeSplatRadiusPx(point.depth);
      const cv::Size outer_axes(
          std::max(1, radius),
          std::max(1, static_cast<int>(std::lround(radius * splat_ellipse_ratio_))));
      BlendFilledEllipse(image, point.pixel, outer_axes, base_color, splat_outer_alpha_);

      if (splat_draw_core_)
      {
        const cv::Scalar core_color(
            Clamp(base_color[0] * 1.08 + 3.0, 0.0, 255.0),
            Clamp(base_color[1] * 1.08 + 3.0, 0.0, 255.0),
            Clamp(base_color[2] * 1.08 + 3.0, 0.0, 255.0));
        const int core_radius = std::max(1, static_cast<int>(std::lround(radius * 0.55)));
        const cv::Size core_axes(
            std::max(1, core_radius),
            std::max(1, static_cast<int>(std::lround(core_radius * splat_ellipse_ratio_ * 0.9))));
        BlendFilledEllipse(image, point.pixel, core_axes, core_color, splat_inner_alpha_);
      }
    }
  }

  cv::Mat RenderFrame(
      const std::vector<RenderPoint> &points, const rclcpp::Time &stamp,
      const rclcpp::Time &render_time) const
  {
    cv::Mat image(height_, width_, CV_8UC3, cv::Scalar(background_b_, background_g_, background_r_));
    if (draw_grid_)
    {
      DrawGrid(image);
    }

    DrawSplats(image, BuildProjectedPoints(points));

    std::ostringstream label;
    label << "Colored cloud view  yaw=" << std::fixed << std::setprecision(1) << yaw_deg_
          << " pitch=" << pitch_deg_
          << " points=" << points.size();
    cv::putText(
        image, label.str(), cv::Point(24, 36), cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(230, 235, 240), 2, cv::LINE_AA);

    if (stamp.nanoseconds() > 0)
    {
      const double age_ms = std::fabs((render_time - stamp).seconds()) * 1000.0;
      std::ostringstream age;
      age << "cloud age " << std::fixed << std::setprecision(0) << age_ms << " ms";
      cv::putText(
          image, age.str(), cv::Point(24, 66), cv::FONT_HERSHEY_SIMPLEX,
          0.55, cv::Scalar(180, 190, 200), 1, cv::LINE_AA);
    }

    return image;
  }

  void OnTimer()
  {
    std::vector<RenderPoint> points;
    rclcpp::Time stamp(0, 0, get_clock()->get_clock_type());
    {
      std::lock_guard<std::mutex> lock(points_mutex_);
      if (!have_cloud_)
      {
        return;
      }
      points = latest_points_;
      stamp = latest_cloud_stamp_;
    }

    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = frame_id_;
    const rclcpp::Time render_time(header.stamp, get_clock()->get_clock_type());
    const double age_ms = std::fabs((render_time - stamp).seconds()) * 1000.0;
    if (stale_cloud_timeout_ms_ > 0 && age_ms > static_cast<double>(stale_cloud_timeout_ms_))
    {
      cv::Mat image(height_, width_, CV_8UC3, cv::Scalar(background_b_, background_g_, background_r_));
      cv::putText(
          image, "waiting for fresh colored cloud", cv::Point(24, 42),
          cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(230, 235, 240), 2, cv::LINE_AA);
      std::ostringstream age;
      age << "last cloud age " << std::fixed << std::setprecision(0) << age_ms << " ms";
      cv::putText(
          image, age.str(), cv::Point(24, 76),
          cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 190, 200), 1, cv::LINE_AA);
      if (publish_enabled_)
      {
        PublishNitrosNv12Image(image, header);
      }
      return;
    }

    const cv::Mat image = RenderFrame(points, stamp, render_time);
    if (publish_enabled_)
    {
      PublishNitrosNv12Image(image, header);
    }
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

  void PublishNitrosNv12Image(const cv::Mat &bgr, const std_msgs::msg::Header &header)
  {
    if (!image_pub_ || bgr.empty())
    {
      return;
    }
    if (!EnsureCudaStream())
    {
      return;
    }

    const int bgr_stride = width_ * 3;
    const int y_stride = width_;
    const int uv_stride = width_ * 2;
    const size_t bgr_size = static_cast<size_t>(bgr_stride) * static_cast<size_t>(height_);
    const size_t nv12_size = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 2U;

    if (!EnsureCudaBuffer(&cuda_bgr_buffer_, cuda_bgr_buffer_size_, bgr_size))
    {
      return;
    }
    if (!publish_pool_ || publish_pool_->size_bytes != nv12_size)
    {
      publish_pool_ = std::make_shared<Nv12PublishPool>(nv12_size);
      publish_pool_->warmup(2);
    }

    void *publish_buffer = publish_pool_->acquire();
    if (!publish_buffer)
    {
      RCLCPP_WARN(get_logger(), "Failed to acquire NV12 publish buffer");
      return;
    }

    cudaError_t err = cudaMemcpyAsync(
        cuda_bgr_buffer_, bgr.data, bgr_size, cudaMemcpyHostToDevice, cuda_stream_);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(get_logger(), "cudaMemcpyAsync HostToDevice failed: %s", cudaGetErrorString(err));
      publish_pool_->release(publish_buffer);
      return;
    }

    err = ConvertBGRToGxfNV12(
        publish_buffer, cuda_bgr_buffer_, width_, height_, y_stride, uv_stride, bgr_stride,
        cuda_stream_);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(get_logger(), "ConvertBGRToGxfNV12 failed: %s", cudaGetErrorString(err));
      publish_pool_->release(publish_buffer);
      return;
    }

    err = cudaStreamSynchronize(cuda_stream_);
    if (err != cudaSuccess)
    {
      RCLCPP_WARN(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
      publish_pool_->release(publish_buffer);
      return;
    }

    try
    {
      auto image = BuildNv12NitrosImageWithPool(
          header, static_cast<uint32_t>(height_), static_cast<uint32_t>(width_),
          publish_buffer, y_stride, uv_stride, publish_pool_);
      image_pub_->publish(image);
    }
    catch (const std::exception &ex)
    {
      RCLCPP_WARN(get_logger(), "Failed to publish rendered cloud image: %s", ex.what());
      publish_pool_->release(publish_buffer);
    }
  }

  std::string cloud_topic_;
  std::string output_topic_;
  std::string frame_id_;
  int width_{1280};
  int height_{720};
  double fps_{20.0};
  double yaw_deg_{90.0};
  double pitch_deg_{-8.0};
  double roll_deg_{0.0};
  double camera_distance_m_{16.0};
  double look_at_x_m_{8.0};
  double look_at_y_m_{0.0};
  double look_at_z_m_{0.3};
  double focal_length_px_{850.0};
  bool publish_enabled_{true};
  int max_points_{200000};
  int splat_radius_min_px_{2};
  int splat_radius_max_px_{7};
  double splat_depth_falloff_m_{18.0};
  double splat_outer_alpha_{0.32};
  double splat_inner_alpha_{0.85};
  double splat_ellipse_ratio_{1.35};
  bool splat_draw_core_{true};
  bool draw_grid_{true};
  double grid_step_m_{1.0};
  double grid_extent_m_{20.0};
  int stale_cloud_timeout_ms_{2000};
  int background_b_{16};
  int background_g_{20};
  int background_r_{24};

  mutable cv::Vec3f camera_position_{};
  mutable cv::Vec3f right_axis_{1.0f, 0.0f, 0.0f};
  mutable cv::Vec3f up_axis_{0.0f, 0.0f, 1.0f};
  mutable cv::Vec3f forward_axis_{0.0f, 1.0f, 0.0f};

  std::mutex points_mutex_;
  std::vector<RenderPoint> latest_points_;
  rclcpp::Time latest_cloud_stamp_{0, 0, RCL_ROS_TIME};
  std::string latest_cloud_frame_id_;
  bool have_cloud_{false};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>>
      image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  void *cuda_bgr_buffer_{nullptr};
  size_t cuda_bgr_buffer_size_{0};
  cudaStream_t cuda_stream_{nullptr};
  std::shared_ptr<Nv12PublishPool> publish_pool_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ColoredCloudViewRenderer>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ColoredCloudViewRenderer)
