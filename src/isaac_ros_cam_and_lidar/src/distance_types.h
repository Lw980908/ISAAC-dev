#pragma once

#include "config.h"

#include <array>
#include <cstdint>
#include <memory>
#include <rclcpp/time.hpp>
#include <vector>

struct CamLidarCalib
{
  enum class CameraModel
  {
    PinholeRational,
    FishEye
  };
  bool valid{false};
  CameraModel model{CameraModel::PinholeRational};
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  std::array<double, 8> dist_pinhole{};
  std::array<double, 4> dist_fisheye{};
  std::array<double, 9> R{};
  std::array<double, 3> t{};
};

struct DistanceEstimationConfig
{
  double tooth_band_height_ratio{0.12};
  double tooth_band_x_margin_ratio{0.08};
  double tooth_band_y_offset_ratio{0.0};
  double near_band_m{0.08};
  double hold_max_dt_ms{250.0};
  double outlier_mad_k{3.5};
  double outlier_min_abs_m{0.02};
  double min_z{0.1};
  double max_z{30.0};
  int min_points{5};
  double max_age_ms{200.0};
};

struct LidarCamPoint
{
  cv::Vec3f cam;
  cv::Vec3f lidar;
};

struct LidarCloudCache
{
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  std::shared_ptr<const std::vector<LidarCamPoint>> points;
};

struct DistanceMetric
{
  Box box{};
  float center_u{0.0f};
  float center_v{0.0f};
  float center_radius_px{0.0f};
  size_t points_in_box{0};
  size_t points_in_center{0};
  size_t near_k{0};
  int roi{0};
  bool has_distance{false};
  float range_m{0.0f};
  float depth_m{0.0f};
  float height_m{0.0f};
};

struct Bbox3DResult
{
  bool valid{false};
  Box source_box{};
  cv::Vec3f center_lidar{0.0f, 0.0f, 0.0f};
  cv::Vec3f size_lidar{0.0f, 0.0f, 0.0f}; // (dx, dy, dz)
  float yaw_rad{0.0f};                    // rotation around +Z in lidar frame

  // 8 corners in lidar frame. Ordering:
  // 0..3: z=min, 4..7: z=max; each face goes (xmin,ymin)->(xmax,ymin)->(xmax,ymax)->(xmin,ymax)
  std::array<cv::Vec3f, 8> corners_lidar{};

  // Projected corners on image (u,v) with validity flags. Only filled when
  // camera intrinsics are available for the request.
  std::array<cv::Point2f, 8> corners_uv{};
  std::array<uint8_t, 8> corners_uv_valid{};
};

struct DistanceBundle
{
  bool calib_valid{false};
  bool cloud_valid{false};
  int64_t cloud_stamp_ns{0};
  double cloud_age_ms{-1.0};
  size_t cloud_points{0};
  std::vector<DistanceMetric> objects;
  std::vector<Bbox3DResult> bbox3d_objects;
  Bbox3DResult bucket_bbox3d{};
};

struct DistanceRequest
{
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  std::vector<Box> boxes;
  bool pending{false};
};
