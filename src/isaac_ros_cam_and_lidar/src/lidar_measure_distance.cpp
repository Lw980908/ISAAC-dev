#include "camera_combined_with_lidar.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <geometry_msgs/msg/point.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <unordered_map>
#include <utility>

namespace
{

  float MedianInPlace(std::vector<float> &values)
  {
    if (values.empty())
    {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float m = values[mid];
    if ((values.size() % 2) == 0)
    {
      const auto it = std::max_element(values.begin(), values.begin() + mid);
      m = 0.5f * (m + *it);
    }
    return m;
  }

  bool ComputeOrientedBboxZ(const std::vector<cv::Vec3f> &pts, cv::Vec3f &center,
                            cv::Vec3f &size, float &yaw_rad,
                            std::array<cv::Vec3f, 8> &corners)
  {
    if (pts.size() < 3)
    {
      return false;
    }

    double mx = 0.0;
    double my = 0.0;
    for (const auto &p : pts)
    {
      mx += static_cast<double>(p[0]);
      my += static_cast<double>(p[1]);
    }
    mx /= static_cast<double>(pts.size());
    my /= static_cast<double>(pts.size());

    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (const auto &p : pts)
    {
      const double dx = static_cast<double>(p[0]) - mx;
      const double dy = static_cast<double>(p[1]) - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    const double inv_n = 1.0 / static_cast<double>(pts.size());
    sxx *= inv_n;
    sxy *= inv_n;
    syy *= inv_n;

    // Largest eigenvector of 2x2 covariance.
    const double tr = sxx + syy;
    const double det = sxx * syy - sxy * sxy;
    const double disc = std::max(0.0, 0.25 * tr * tr - det);
    const double root = std::sqrt(disc);
    const double lambda1 = 0.5 * tr + root;

    double vx = 1.0;
    double vy = 0.0;
    if (std::fabs(sxy) > 1e-9)
    {
      vx = lambda1 - syy;
      vy = sxy;
    }
    else
    {
      vx = (sxx >= syy) ? 1.0 : 0.0;
      vy = (sxx >= syy) ? 0.0 : 1.0;
    }
    const double vnorm = std::sqrt(vx * vx + vy * vy);
    if (!(vnorm > 1e-12))
    {
      return false;
    }
    vx /= vnorm;
    vy /= vnorm;
    yaw_rad = static_cast<float>(std::atan2(vy, vx));

    const double c = std::cos(static_cast<double>(yaw_rad));
    const double s = std::sin(static_cast<double>(yaw_rad));

    double min_xr = std::numeric_limits<double>::infinity();
    double max_xr = -std::numeric_limits<double>::infinity();
    double min_yr = std::numeric_limits<double>::infinity();
    double max_yr = -std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    for (const auto &p : pts)
    {
      const double dx = static_cast<double>(p[0]) - mx;
      const double dy = static_cast<double>(p[1]) - my;
      // Rotate by -yaw: [c s; -s c]
      const double xr = c * dx + s * dy;
      const double yr = -s * dx + c * dy;
      min_xr = std::min(min_xr, xr);
      max_xr = std::max(max_xr, xr);
      min_yr = std::min(min_yr, yr);
      max_yr = std::max(max_yr, yr);
      const double z = static_cast<double>(p[2]);
      min_z = std::min(min_z, z);
      max_z = std::max(max_z, z);
    }

    const double cxr = 0.5 * (min_xr + max_xr);
    const double cyr = 0.5 * (min_yr + max_yr);
    const double cz = 0.5 * (min_z + max_z);
    const double dx = std::max(0.0, max_xr - min_xr);
    const double dy = std::max(0.0, max_yr - min_yr);
    const double dz = std::max(0.0, max_z - min_z);

    const double cx = mx + (c * cxr - s * cyr);
    const double cy = my + (s * cxr + c * cyr);

    center = cv::Vec3f(static_cast<float>(cx), static_cast<float>(cy),
                       static_cast<float>(cz));
    size = cv::Vec3f(static_cast<float>(dx), static_cast<float>(dy),
                     static_cast<float>(dz));

    const double hx = 0.5 * dx;
    const double hy = 0.5 * dy;
    const double hz = 0.5 * dz;
    const double local_x[2] = {-hx, hx};
    const double local_y[2] = {-hy, hy};
    const double local_z[2] = {-hz, hz};

    int idx = 0;
    for (int iz = 0; iz < 2; ++iz)
    {
      for (int iy = 0; iy < 2; ++iy)
      {
        for (int ix = 0; ix < 2; ++ix)
        {
          const double lx = local_x[ix];
          const double ly = local_y[iy];
          const double xw = (c * lx - s * ly) + cx;
          const double yw = (s * lx + c * ly) + cy;
          const double zw = static_cast<double>(center[2]) + local_z[iz];
          corners[static_cast<size_t>(idx)] = cv::Vec3f(
              static_cast<float>(xw), static_cast<float>(yw), static_cast<float>(zw));
          idx++;
        }
      }
    }
    // Reorder to (xmin,ymin)->(xmax,ymin)->(xmax,ymax)->(xmin,ymax) for each z.
    // Current order is x fastest, then y, then z: (x-,y-),(x+,y-),(x-,y+),(x+,y+)
    // Swap indices 2 and 3 within each z plane.
    for (int z = 0; z < 2; ++z)
    {
      const int base = z * 4;
      std::swap(corners[static_cast<size_t>(base + 2)],
                corners[static_cast<size_t>(base + 3)]);
    }
    return true;
  }

  float IoU2D(const Box &a, const Box &b)
  {
    const float ax0 = std::min(a.left, a.right);
    const float ay0 = std::min(a.top, a.bottom);
    const float ax1 = std::max(a.left, a.right);
    const float ay1 = std::max(a.top, a.bottom);
    const float bx0 = std::min(b.left, b.right);
    const float by0 = std::min(b.top, b.bottom);
    const float bx1 = std::max(b.left, b.right);
    const float by1 = std::max(b.top, b.bottom);

    const float ix0 = std::max(ax0, bx0);
    const float iy0 = std::max(ay0, by0);
    const float ix1 = std::min(ax1, bx1);
    const float iy1 = std::min(ay1, by1);
    const float iw = std::max(0.0f, ix1 - ix0);
    const float ih = std::max(0.0f, iy1 - iy0);
    const float inter = iw * ih;

    const float area_a = std::max(0.0f, ax1 - ax0) * std::max(0.0f, ay1 - ay0);
    const float area_b = std::max(0.0f, bx1 - bx0) * std::max(0.0f, by1 - by0);
    const float uni = area_a + area_b - inter;
    if (!(uni > 1e-6f))
    {
      return 0.0f;
    }
    return inter / uni;
  }

  constexpr std::array<std::pair<int, int>, 12> kBboxEdges = {
      std::pair<int, int>{0, 1}, std::pair<int, int>{1, 2},
      std::pair<int, int>{2, 3}, std::pair<int, int>{3, 0},
      std::pair<int, int>{4, 5}, std::pair<int, int>{5, 6},
      std::pair<int, int>{6, 7}, std::pair<int, int>{7, 4},
      std::pair<int, int>{0, 4}, std::pair<int, int>{1, 5},
      std::pair<int, int>{2, 6}, std::pair<int, int>{3, 7}};

} // namespace

void CameraCombinedWithLidarPipelineManager::handleLivoxCustomMsg(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }
  if (!cam_lidar_calib_.valid)
  {
    return;
  }

  std::vector<LidarCamPoint> pts;
  pts.reserve(msg->points.size());

  const auto &R = cam_lidar_calib_.R;
  const auto &t = cam_lidar_calib_.t;
  for (const auto &p : msg->points)
  {
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    {
      continue;
    }
    const float X = static_cast<float>(R[0] * x + R[1] * y + R[2] * z + t[0]);
    const float Y = static_cast<float>(R[3] * x + R[4] * y + R[5] * z + t[1]);
    const float Z = static_cast<float>(R[6] * x + R[7] * y + R[8] * z + t[2]);
    pts.push_back(LidarCamPoint{cv::Vec3f(X, Y, Z), cv::Vec3f(x, y, z)});
  }

  auto ptr = std::make_shared<const std::vector<LidarCamPoint>>(std::move(pts));
  const rclcpp::Time stamp(msg->header.stamp);
  {
    std::lock_guard<std::mutex> lk(lidar_cloud_mutex_);
    lidar_cloud_cache_.stamp = stamp;
    lidar_cloud_cache_.points = ptr;
  }
  {
    std::lock_guard<std::mutex> hk(lidar_cloud_history_mutex_);
    last_lidar_frame_id_ = msg->header.frame_id;
    lidar_cloud_history_.push_back(LidarCloudFrame{stamp, ptr});

    // Prune by age and max frames.
    const int64_t win_ns =
        static_cast<int64_t>(std::llround(bbox3d_cfg_.window_ms * 1000000.0));
    const auto newest_ns = stamp.nanoseconds();
    while (lidar_cloud_history_.size() >
           static_cast<size_t>(std::max(1, bbox3d_cfg_.max_frames)))
    {
      lidar_cloud_history_.pop_front();
    }
    while (!lidar_cloud_history_.empty() && win_ns > 0 &&
           (newest_ns - lidar_cloud_history_.front().stamp.nanoseconds()) >
               win_ns)
    {
      lidar_cloud_history_.pop_front();
    }
  }
}

void CameraCombinedWithLidarPipelineManager::handlePointCloud2(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }

  std::vector<LidarCamPoint> pts;
  pts.reserve(static_cast<size_t>(msg->width) *
              static_cast<size_t>(msg->height));

  const auto &R = cam_lidar_calib_.R;
  const auto &t = cam_lidar_calib_.t;
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
      const float X = static_cast<float>(R[0] * x + R[1] * y + R[2] * z + t[0]);
      const float Y = static_cast<float>(R[3] * x + R[4] * y + R[5] * z + t[1]);
      const float Z = static_cast<float>(R[6] * x + R[7] * y + R[8] * z + t[2]);
      pts.push_back(LidarCamPoint{cv::Vec3f(X, Y, Z), cv::Vec3f(x, y, z)});
    }
  }
  catch (...)
  {
    return;
  }

  auto ptr = std::make_shared<const std::vector<LidarCamPoint>>(std::move(pts));
  const rclcpp::Time stamp(msg->header.stamp);
  {
    std::lock_guard<std::mutex> lk(lidar_cloud_mutex_);
    lidar_cloud_cache_.stamp = stamp;
    lidar_cloud_cache_.points = ptr;
  }
  {
    std::lock_guard<std::mutex> hk(lidar_cloud_history_mutex_);
    last_lidar_frame_id_ = msg->header.frame_id;
    lidar_cloud_history_.push_back(LidarCloudFrame{stamp, ptr});

    const int64_t win_ns =
        static_cast<int64_t>(std::llround(bbox3d_cfg_.window_ms * 1000000.0));
    const auto newest_ns = stamp.nanoseconds();
    while (lidar_cloud_history_.size() >
           static_cast<size_t>(std::max(1, bbox3d_cfg_.max_frames)))
    {
      lidar_cloud_history_.pop_front();
    }
    while (!lidar_cloud_history_.empty() && win_ns > 0 &&
           (newest_ns - lidar_cloud_history_.front().stamp.nanoseconds()) >
               win_ns)
    {
      lidar_cloud_history_.pop_front();
    }
  }
}

DistanceBundle CameraCombinedWithLidarPipelineManager::computeDistanceBundle(
    const rclcpp::Time &stamp, const DistanceRequest &req)
{
  const auto &boxes = req.boxes;
  DistanceBundle bundle;
  bundle.calib_valid = cam_lidar_calib_.valid;

  struct SmoothState
  {
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    float range_m{0.0f};
    float depth_m{0.0f};
    float height_m{0.0f};
    bool valid{false};
  };
  static std::mutex smooth_mutex;
  static std::unordered_map<uint64_t, SmoothState> smooth_states;

  rclcpp::Time cloud_stamp(0, 0, RCL_ROS_TIME);
  std::shared_ptr<const std::vector<LidarCamPoint>> cloud_pts;
  {
    std::lock_guard<std::mutex> lk(lidar_cloud_mutex_);
    cloud_stamp = lidar_cloud_cache_.stamp;
    cloud_pts = lidar_cloud_cache_.points;
  }

  const int64_t dt_ns =
      cloud_pts ? (stamp.nanoseconds() - cloud_stamp.nanoseconds()) : 0;
  const double age_ms =
      cloud_pts ? (static_cast<double>(std::llabs(dt_ns)) / 1000000.0) : -1.0;
  bundle.cloud_age_ms = age_ms;
  bundle.cloud_stamp_ns = cloud_stamp.nanoseconds();
  bundle.cloud_points = cloud_pts ? cloud_pts->size() : 0;
  bundle.cloud_valid =
      static_cast<bool>(cloud_pts) && (age_ms <= distance_cfg_.max_age_ms);
  if (!bundle.cloud_valid)
  {
    cloud_pts.reset();
    bundle.cloud_stamp_ns = 0;
    bundle.cloud_points = 0;
  }

  double fx = cam_lidar_calib_.fx;
  double fy = cam_lidar_calib_.fy;
  double cx = cam_lidar_calib_.cx;
  double cy = cam_lidar_calib_.cy;
  if (!(fx > 0.0 && fy > 0.0))
  {
    const auto &uc = undistort_config_;
    const size_t idx = static_cast<size_t>(std::max(0, input_config_.idx));
    const size_t safe_idx =
        uc.camera_type_groups.empty()
            ? static_cast<size_t>(0)
            : std::min(idx, uc.camera_type_groups.size() - 1);
    const auto ct = uc.camera_type_groups.empty()
                        ? CameraType::Fish_eye_camera
                        : uc.camera_type_groups[safe_idx];
    if (ct == CameraType::Pin_hole_camera &&
        !uc.Pin_hole_param_groups.empty())
    {
      const size_t pi = std::min(safe_idx, uc.Pin_hole_param_groups.size() - 1);
      fx = uc.Pin_hole_param_groups[pi].fx;
      fy = uc.Pin_hole_param_groups[pi].fy;
      cx = uc.Pin_hole_param_groups[pi].cx;
      cy = uc.Pin_hole_param_groups[pi].cy;
    }
    else if (!uc.Fisheye_param_groups.empty())
    {
      const size_t fi = std::min(safe_idx, uc.Fisheye_param_groups.size() - 1);
      fx = uc.Fisheye_param_groups[fi].fx;
      fy = uc.Fisheye_param_groups[fi].fy;
      cx = uc.Fisheye_param_groups[fi].cx;
      cy = uc.Fisheye_param_groups[fi].cy;
    }
  }

  const auto cam_model = cam_lidar_calib_.model;
  const auto dist_ph = cam_lidar_calib_.dist_pinhole;
  const auto dist_fe = cam_lidar_calib_.dist_fisheye;
  auto project_uv = [&](float X, float Y, float Z, float &u, float &v) -> bool
  {
    if (!(Z > 0.0f) || !(fx > 0.0 && fy > 0.0))
    {
      return false;
    }
    const double x = static_cast<double>(X) / static_cast<double>(Z);
    const double y = static_cast<double>(Y) / static_cast<double>(Z);

    if (cam_model == CamLidarCalib::CameraModel::FishEye)
    {
      const double k1 = dist_fe[0];
      const double k2 = dist_fe[1];
      const double k3 = dist_fe[2];
      const double k4 = dist_fe[3];
      const double r = std::sqrt(x * x + y * y);
      double xd = x;
      double yd = y;
      if (r > 1e-12)
      {
        const double theta = std::atan(r);
        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta4 * theta4;
        const double theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 +
                                        k3 * theta6 + k4 * theta8);
        const double scale = theta_d / r;
        xd = x * scale;
        yd = y * scale;
      }
      u = static_cast<float>(fx * xd + cx);
      v = static_cast<float>(fy * yd + cy);
      return std::isfinite(u) && std::isfinite(v);
    }

    const double k1 = dist_ph[0];
    const double k2 = dist_ph[1];
    const double p1 = dist_ph[2];
    const double p2 = dist_ph[3];
    const double k3 = dist_ph[4];
    const double k4 = dist_ph[5];
    const double k5 = dist_ph[6];
    const double k6 = dist_ph[7];

    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
    if (std::fabs(denom) > 1e-12)
    {
      radial /= denom;
    }

    const double x2 = x * x;
    const double y2 = y * y;
    const double xy = x * y;
    const double x_tang = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
    const double y_tang = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy;

    const double xd = x * radial + x_tang;
    const double yd = y * radial + y_tang;
    u = static_cast<float>(fx * xd + cx);
    v = static_cast<float>(fy * yd + cy);
    return std::isfinite(u) && std::isfinite(v);
  };

  bundle.objects.reserve(boxes.size());
  bundle.bbox3d_objects.reserve(boxes.size());

  const float kMinCamZ = static_cast<float>(distance_cfg_.min_z);
  const float kMaxCamZ = static_cast<float>(distance_cfg_.max_z);
  const size_t min_pts =
      static_cast<size_t>(std::max(1, distance_cfg_.min_points));

  struct Candidate
  {
    float sort_key;
    float range;
    float depth;
    float height;
  };

  static thread_local std::vector<Candidate> box_candidates;
  static thread_local std::vector<Candidate> tooth_candidates;
  static thread_local std::vector<Candidate> tooth_candidates_wide;
  static thread_local std::vector<float> ranges;
  static thread_local std::vector<float> depths;
  static thread_local std::vector<float> heights;
  static thread_local std::vector<float> tmp_depths;
  static thread_local std::vector<float> tmp_absdevs;

  for (const auto &b : boxes)
  {
    if (b.label >= 2 && b.label <= 10)
    {
      continue;
    }

    DistanceMetric m;
    m.box = b;

    const float l = std::min(b.left, b.right);
    const float r = std::max(b.left, b.right);
    const float t = std::min(b.top, b.bottom);
    const float bb = std::max(b.top, b.bottom);

    const float bw = std::max(0.0f, r - l);
    const float bh = std::max(0.0f, bb - t);

    const double band_h_ratio =
        std::max(0.01, std::min(1.0, distance_cfg_.tooth_band_height_ratio));
    const double band_x_margin_ratio =
        std::max(0.0, std::min(0.45, distance_cfg_.tooth_band_x_margin_ratio));
    const double band_y_offset_ratio =
        std::max(0.0, std::min(0.45, distance_cfg_.tooth_band_y_offset_ratio));

    const double band_h_ratio_wide =
        std::max(band_h_ratio, std::min(0.35, band_h_ratio * 2.0));
    const double band_x_margin_ratio_wide =
        std::max(0.0, std::min(0.45, band_x_margin_ratio * 0.5));

    const float band_h = std::max(
        2.0f, static_cast<float>(static_cast<double>(bh) * band_h_ratio));
    const float y_offset =
        static_cast<float>(static_cast<double>(bh) * band_y_offset_ratio);
    const float band_bottom = std::max(t, std::min(bb, bb - y_offset));
    const float band_top = std::max(t, band_bottom - band_h);

    const float band_h_wide = std::max(
        2.0f, static_cast<float>(static_cast<double>(bh) * band_h_ratio_wide));
    const float band_bottom_wide = band_bottom;
    const float band_top_wide = std::max(t, band_bottom_wide - band_h_wide);

    const float x_margin =
        static_cast<float>(static_cast<double>(bw) * band_x_margin_ratio);
    float band_l = l + x_margin;
    float band_r = r - x_margin;
    if (!(band_l < band_r))
    {
      band_l = l;
      band_r = r;
    }

    const float x_margin_wide =
        static_cast<float>(static_cast<double>(bw) * band_x_margin_ratio_wide);
    float band_l_wide = l + x_margin_wide;
    float band_r_wide = r - x_margin_wide;
    if (!(band_l_wide < band_r_wide))
    {
      band_l_wide = l;
      band_r_wide = r;
    }

    const float u_ref = 0.5f * (band_l + band_r);
    const float v_ref = 0.5f * (band_top + band_bottom);
    const float u_ref_wide = 0.5f * (band_l_wide + band_r_wide);
    const float v_ref_wide = 0.5f * (band_top_wide + band_bottom_wide);

    m.center_u = u_ref;
    m.center_v = v_ref;
    m.center_radius_px = band_h;

    if (!bundle.calib_valid || !cloud_pts)
    {
      bundle.objects.push_back(std::move(m));
      continue;
    }

    box_candidates.clear();
    tooth_candidates.clear();
    tooth_candidates_wide.clear();
    if (box_candidates.capacity() < 128)
    {
      box_candidates.reserve(128);
    }
    if (tooth_candidates.capacity() < 64)
    {
      tooth_candidates.reserve(64);
    }
    if (tooth_candidates_wide.capacity() < 96)
    {
      tooth_candidates_wide.reserve(96);
    }

    for (const auto &p : *cloud_pts)
    {
      const float X = p.cam[0];
      const float Y = p.cam[1];
      const float Z = p.cam[2];
      const float lx = p.lidar[0];
      const float ly = p.lidar[1];
      const float lz = p.lidar[2];
      if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z) ||
          !std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz))
      {
        continue;
      }
      if (Z <= kMinCamZ || Z >= kMaxCamZ)
      {
        continue;
      }
      float u = 0.0f;
      float v = 0.0f;
      if (!project_uv(X, Y, Z, u, v))
      {
        continue;
      }
      if (u < l || u > r || v < t || v > bb)
      {
        continue;
      }
      const float forward_lidar = lx;
      if (!(forward_lidar > 0.0f))
      {
        continue;
      }
      const float range_lidar = std::sqrt(lx * lx + ly * ly + lz * lz);
      if (!(range_lidar > 0.0f))
      {
        continue;
      }
      const Candidate c{forward_lidar, range_lidar, forward_lidar, lz};
      box_candidates.push_back(c);
      if (u >= band_l && u <= band_r && v >= band_top && v <= band_bottom)
      {
        tooth_candidates.push_back(c);
      }
      if (u >= band_l_wide && u <= band_r_wide && v >= band_top_wide &&
          v <= band_bottom_wide)
      {
        tooth_candidates_wide.push_back(c);
      }
    }

    m.points_in_box = box_candidates.size();
    m.points_in_center = tooth_candidates.size();

    auto reject_outliers_inplace = [&](size_t required) -> bool
    {
      const double mad_k = distance_cfg_.outlier_mad_k;
      const double min_abs = distance_cfg_.outlier_min_abs_m;
      if (!(mad_k > 0.0) || depths.size() < required)
      {
        return depths.size() >= required;
      }
      tmp_depths = depths;
      const float med = MedianInPlace(tmp_depths);
      tmp_absdevs.resize(depths.size());
      for (size_t i = 0; i < depths.size(); ++i)
      {
        tmp_absdevs[i] = std::fabs(depths[i] - med);
      }
      tmp_depths = tmp_absdevs;
      const float mad = MedianInPlace(tmp_depths);
      const float th = std::max(static_cast<float>(min_abs),
                                static_cast<float>(mad_k) * mad);
      size_t w = 0;
      for (size_t i = 0; i < depths.size(); ++i)
      {
        if (tmp_absdevs[i] <= th)
        {
          ranges[w] = ranges[i];
          depths[w] = depths[i];
          heights[w] = heights[i];
          w++;
        }
      }
      ranges.resize(w);
      depths.resize(w);
      heights.resize(w);
      return depths.size() >= required;
    };

    float box_ref_depth = 0.0f;
    bool have_box_ref = false;
    const float band_m =
        std::max(0.01f, static_cast<float>(distance_cfg_.near_band_m));
    if (box_candidates.size() >= min_pts)
    {
      tmp_depths.resize(box_candidates.size());
      for (size_t i = 0; i < box_candidates.size(); ++i)
      {
        tmp_depths[i] = box_candidates[i].depth;
      }
      const size_t p = std::min(tmp_depths.size() - 1, tmp_depths.size() / 10);
      if (p < tmp_depths.size())
      {
        std::nth_element(tmp_depths.begin(),
                         tmp_depths.begin() + static_cast<std::ptrdiff_t>(p),
                         tmp_depths.end());
      }
      const float p10 = tmp_depths[p];
      const float gate = p10 + band_m;

      ranges.clear();
      depths.clear();
      heights.clear();
      for (const auto &c : box_candidates)
      {
        if (c.depth <= gate)
        {
          ranges.push_back(c.range);
          depths.push_back(c.depth);
          heights.push_back(c.height);
        }
      }
      if (reject_outliers_inplace(min_pts))
      {
        tmp_depths = depths;
        box_ref_depth = MedianInPlace(tmp_depths);
        have_box_ref = std::isfinite(box_ref_depth) && (box_ref_depth > 0.0f);
      }
    }
    const float *box_ref_ptr = have_box_ref ? &box_ref_depth : nullptr;

    auto estimate_from = [&](std::vector<Candidate> &cands, int roi,
                             size_t required_min_pts,
                             const float *ref_depth) -> bool
    {
      const size_t n = cands.size();
      if (n < required_min_pts)
      {
        return false;
      }

      float base = 0.0f;
      if (ref_depth)
      {
        base = *ref_depth;
      }
      else
      {
        float min_key = std::numeric_limits<float>::infinity();
        for (const auto &c : cands)
        {
          if (c.sort_key < min_key)
          {
            min_key = c.sort_key;
          }
        }
        if (!std::isfinite(min_key))
        {
          return false;
        }
        base = min_key;
      }

      const float gate = base + band_m;
      ranges.clear();
      depths.clear();
      heights.clear();
      for (const auto &c : cands)
      {
        if (c.depth <= gate)
        {
          ranges.push_back(c.range);
          depths.push_back(c.depth);
          heights.push_back(c.height);
        }
      }
      if (depths.size() < required_min_pts)
      {
        return false;
      }
      if (!reject_outliers_inplace(required_min_pts))
      {
        return false;
      }

      m.roi = roi;
      m.near_k = depths.size();
      m.range_m = MedianInPlace(ranges);
      m.depth_m = MedianInPlace(depths);
      m.height_m = MedianInPlace(heights);
      m.has_distance = true;
      return true;
    };

    const size_t min_pts_tooth = min_pts;
    const int key_cx =
        static_cast<int>(std::lround((static_cast<double>(l) + r) * 0.5 / 32.0));
    const int key_cy =
        static_cast<int>(std::lround((static_cast<double>(t) + bb) * 0.5 / 32.0));
    const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint16_t>(m.box.label & 0xFFFF)) << 32) |
        (static_cast<uint64_t>(static_cast<uint16_t>(key_cx & 0xFFFF)) << 16) |
        static_cast<uint64_t>(static_cast<uint16_t>(key_cy & 0xFFFF));
    bool have_prev_depth = false;
    {
      std::lock_guard<std::mutex> lk(smooth_mutex);
      auto it = smooth_states.find(key);
      if (it != smooth_states.end() && it->second.valid)
      {
        have_prev_depth =
            std::isfinite(it->second.depth_m) && (it->second.depth_m > 0.0f);
      }
    }

    bool used_tooth_wide = false;
    if (estimate_from(tooth_candidates, 1, min_pts_tooth, box_ref_ptr))
    {
    }
    else if (estimate_from(tooth_candidates_wide, 1, min_pts_tooth,
                           box_ref_ptr))
    {
      used_tooth_wide = true;
    }
    else
    {
      m.roi = 0;
      m.near_k = 0;
      m.has_distance = false;
    }

    if (m.roi == 1)
    {
      if (used_tooth_wide)
      {
        m.center_u = u_ref_wide;
        m.center_v = v_ref_wide;
        m.center_radius_px = band_h_wide;
        m.points_in_center = tooth_candidates_wide.size();
      }
      else
      {
        m.center_u = u_ref;
        m.center_v = v_ref;
        m.center_radius_px = band_h;
        m.points_in_center = tooth_candidates.size();
      }
    }

    if (!m.has_distance && have_prev_depth)
    {
      const double hold_ms = distance_cfg_.hold_max_dt_ms;
      if (hold_ms > 0.0)
      {
        SmoothState prev;
        bool have_prev = false;
        {
          std::lock_guard<std::mutex> lk(smooth_mutex);
          auto it = smooth_states.find(key);
          if (it != smooth_states.end() && it->second.valid)
          {
            prev = it->second;
            have_prev = true;
          }
        }
        if (have_prev)
        {
          const int64_t dt_ns = stamp.nanoseconds() - prev.stamp.nanoseconds();
          const double dt_ms =
              static_cast<double>(std::llabs(dt_ns)) / 1000000.0;
          if (dt_ms <= hold_ms)
          {
            m.roi = 3;
            m.near_k = 0;
            m.range_m = prev.range_m;
            m.depth_m = prev.depth_m;
            m.height_m = prev.height_m;
            m.has_distance = true;
            m.center_u = u_ref;
            m.center_v = v_ref;
            m.center_radius_px = band_h;
            m.points_in_center = tooth_candidates.size();
          }
        }
      }
    }

    if (m.has_distance)
    {
      SmoothState next;
      next.stamp = stamp;
      next.range_m = m.range_m;
      next.depth_m = m.depth_m;
      next.height_m = m.height_m;
      next.valid = true;
      {
        std::lock_guard<std::mutex> lk(smooth_mutex);
        smooth_states[key] = next;
      }
    }

    bundle.objects.push_back(std::move(m));
  }

  if (use_nitros_sub_ && bbox3d_cfg_.enable && cam_lidar_calib_.valid)
  {
    std::vector<std::shared_ptr<const std::vector<LidarCamPoint>>> frames;
    frames.reserve(static_cast<size_t>(std::max(1, bbox3d_cfg_.max_frames)));
    rclcpp::Time newest_hist_stamp{0, 0, RCL_ROS_TIME};
    bool have_hist_stamp = false;
    {
      std::lock_guard<std::mutex> lk(lidar_cloud_history_mutex_);
      if (!lidar_cloud_history_.empty())
      {
        newest_hist_stamp = lidar_cloud_history_.back().stamp;
        have_hist_stamp = true;
      }
      for (const auto &f : lidar_cloud_history_)
      {
        if (f.points)
        {
          frames.push_back(f.points);
        }
      }
    }
    if (have_hist_stamp)
    {
      const int64_t dt_hist_ns =
          stamp.nanoseconds() - newest_hist_stamp.nanoseconds();
      const double age_hist_ms =
          static_cast<double>(std::llabs(dt_hist_ns)) / 1000000.0;
      if (bbox3d_cfg_.max_age_ms > 0.0 && age_hist_ms > bbox3d_cfg_.max_age_ms)
      {
        frames.clear();
      }
    }
    if (frames.empty())
    {
      std::shared_ptr<const std::vector<LidarCamPoint>> cache_pts;
      {
        std::lock_guard<std::mutex> lk(lidar_cloud_mutex_);
        cache_pts = lidar_cloud_cache_.points;
      }
      if (cache_pts)
      {
        frames.push_back(cache_pts);
      }
      else if (cloud_pts)
      {
        frames.push_back(cloud_pts);
      }
    }

    size_t total_pts = 0;
    for (const auto &p : frames)
    {
      total_pts += p ? p->size() : 0;
    }
    const size_t max_pts =
        static_cast<size_t>(std::max(1000, bbox3d_cfg_.max_points));
    const size_t stride = (total_pts > max_pts) ? (total_pts / max_pts + 1) : 1;
    const size_t min_bbox_pts =
        static_cast<size_t>(std::max(3, bbox3d_cfg_.min_points));

    struct RawBboxPt
    {
      cv::Vec3f lidar;
      float depth_lidar_x{0.0f};
    };
    static thread_local std::vector<RawBboxPt> bbox_pts_raw;
    static thread_local std::vector<cv::Vec3f> bbox_pts;
    static thread_local std::vector<float> bbox_depths;
    static thread_local std::vector<int> hist;
    static thread_local std::vector<float> xs;
    static thread_local std::vector<float> ys;
    static thread_local std::vector<float> zs;
    static thread_local std::vector<float> absdev;

    const auto project_bbox3d_corners = [&](Bbox3DResult &bbox3d)
    {
      bbox3d.corners_uv.fill(cv::Point2f(-1.0f, -1.0f));
      bbox3d.corners_uv_valid.fill(0);

      const auto &R = cam_lidar_calib_.R;
      const auto &t = cam_lidar_calib_.t;
      for (size_t i = 0; i < bbox3d.corners_lidar.size(); ++i)
      {
        const auto &corner = bbox3d.corners_lidar[i];
        const float X = static_cast<float>(R[0] * corner[0] + R[1] * corner[1] +
                                           R[2] * corner[2] + t[0]);
        const float Y = static_cast<float>(R[3] * corner[0] + R[4] * corner[1] +
                                           R[5] * corner[2] + t[1]);
        const float Z = static_cast<float>(R[6] * corner[0] + R[7] * corner[1] +
                                           R[8] * corner[2] + t[2]);
        float u = 0.0f;
        float v = 0.0f;
        if (project_uv(X, Y, Z, u, v))
        {
          bbox3d.corners_uv[i] = cv::Point2f(u, v);
          bbox3d.corners_uv_valid[i] = 1;
        }
      }
    };

    const auto find_depth_reference = [&](const Box &box) -> float
    {
      float best_iou = 0.0f;
      float depth_ref = std::numeric_limits<float>::quiet_NaN();
      for (const auto &m : bundle.objects)
      {
        if (!m.has_distance)
        {
          continue;
        }
        const float iou = IoU2D(m.box, box);
        if (iou > best_iou)
        {
          best_iou = iou;
          depth_ref = m.depth_m;
        }
      }
      return depth_ref;
    };

    const auto try_fit_bbox3d = [&](const Box &box, Bbox3DResult &out) -> bool
    {
      const float l = std::min(box.left, box.right);
      const float r = std::max(box.left, box.right);
      const float t = std::min(box.top, box.bottom);
      const float b = std::max(box.top, box.bottom);
      if (!(r > l && b > t))
      {
        return false;
      }

      const float bw = std::max(1.0f, r - l);
      const float bh = std::max(1.0f, b - t);
      const float shrink = std::max(
          0.0f, std::min(0.45f, static_cast<float>(bbox3d_cfg_.roi_shrink_ratio)));
      float x0 = l + shrink * bw;
      float y0 = t + shrink * bh;
      float x1 = r - shrink * bw;
      float y1 = b - shrink * bh;
      if (!(x1 > x0 && y1 > y0))
      {
        x0 = l;
        y0 = t;
        x1 = r;
        y1 = b;
      }

      bbox_pts_raw.clear();
      bbox_pts_raw.reserve(min_bbox_pts * 2);

      size_t global_idx = 0;
      for (const auto &f : frames)
      {
        if (!f)
        {
          continue;
        }
        for (const auto &pt : *f)
        {
          global_idx++;
          if (stride > 1 && (global_idx % stride) != 0)
          {
            continue;
          }
          const float Z = pt.cam[2];
          if (!(Z > kMinCamZ && Z < kMaxCamZ))
          {
            continue;
          }
          float u = 0.0f;
          float v = 0.0f;
          if (!project_uv(pt.cam[0], pt.cam[1], pt.cam[2], u, v))
          {
            continue;
          }
          if (!(u >= x0 && u <= x1 && v >= y0 && v <= y1))
          {
            continue;
          }
          if (!(pt.lidar[0] > 0.0f))
          {
            continue;
          }
          bbox_pts_raw.push_back(RawBboxPt{pt.lidar, pt.lidar[0]});
        }
      }

      if (bbox_pts_raw.size() < min_bbox_pts)
      {
        return false;
      }

      float depth_ref = find_depth_reference(box);
      if (!std::isfinite(depth_ref))
      {
        bbox_depths.resize(bbox_pts_raw.size());
        for (size_t i = 0; i < bbox_pts_raw.size(); ++i)
        {
          bbox_depths[i] = bbox_pts_raw[i].depth_lidar_x;
        }
        const size_t p =
            std::min(bbox_depths.size() - 1, bbox_depths.size() / 10);
        std::nth_element(bbox_depths.begin(),
                         bbox_depths.begin() + static_cast<std::ptrdiff_t>(p),
                         bbox_depths.end());
        depth_ref = bbox_depths[p];
      }

      if (std::isfinite(depth_ref) && bbox3d_cfg_.depth_gate_m > 0.0)
      {
        const float gate = static_cast<float>(bbox3d_cfg_.depth_gate_m);
        bbox_pts_raw.erase(
            std::remove_if(bbox_pts_raw.begin(), bbox_pts_raw.end(),
                           [&](const RawBboxPt &p)
                           {
                             return std::fabs(p.depth_lidar_x - depth_ref) > gate;
                           }),
            bbox_pts_raw.end());
      }
      if (bbox_pts_raw.size() < min_bbox_pts)
      {
        return false;
      }

      const float bin =
          static_cast<float>(std::max(0.05, bbox3d_cfg_.depth_bin_m));
      float dmin = std::numeric_limits<float>::infinity();
      float dmax = -std::numeric_limits<float>::infinity();
      for (const auto &p : bbox_pts_raw)
      {
        dmin = std::min(dmin, p.depth_lidar_x);
        dmax = std::max(dmax, p.depth_lidar_x);
      }
      const int nb = std::max(
          1, static_cast<int>(std::ceil(std::max(0.0f, dmax - dmin) / bin)));
      hist.assign(static_cast<size_t>(nb), 0);
      for (const auto &p : bbox_pts_raw)
      {
        const int k = std::max(
            0, std::min(nb - 1, static_cast<int>((p.depth_lidar_x - dmin) / bin)));
        hist[static_cast<size_t>(k)] += 1;
      }

      const int min_cnt =
          std::max(5, static_cast<int>(bbox3d_cfg_.min_points) / 20);
      const auto bin_center = [&](int k)
      { return dmin + (static_cast<float>(k) + 0.5f) * bin; };

      int best_k = 0;
      if (std::isfinite(depth_ref))
      {
        float best_err = std::numeric_limits<float>::infinity();
        for (int k = 0; k < nb; ++k)
        {
          if (hist[static_cast<size_t>(k)] < 1)
          {
            continue;
          }
          const float err = std::fabs(bin_center(k) - depth_ref);
          const bool enough = hist[static_cast<size_t>(k)] >= min_cnt;
          if (enough && err < best_err)
          {
            best_err = err;
            best_k = k;
          }
        }
        if (!(best_err < std::numeric_limits<float>::infinity()))
        {
          for (int k = 0; k < nb; ++k)
          {
            if (hist[static_cast<size_t>(k)] < 1)
            {
              continue;
            }
            const float err = std::fabs(bin_center(k) - depth_ref);
            if (err < best_err)
            {
              best_err = err;
              best_k = k;
            }
          }
        }
      }
      else
      {
        bool found = false;
        for (int k = 0; k < nb; ++k)
        {
          if (hist[static_cast<size_t>(k)] >= min_cnt)
          {
            best_k = k;
            found = true;
            break;
          }
        }
        if (!found)
        {
          int best_cnt = hist[0];
          for (int k = 1; k < nb; ++k)
          {
            if (hist[static_cast<size_t>(k)] > best_cnt)
            {
              best_cnt = hist[static_cast<size_t>(k)];
              best_k = k;
            }
          }
        }
      }

      const float lo = dmin + static_cast<float>(best_k) * bin - 0.5f * bin;
      const float hi = dmin + static_cast<float>(best_k + 1) * bin + 0.5f * bin;
      bbox_pts.clear();
      bbox_pts.reserve(bbox_pts_raw.size());
      for (const auto &p : bbox_pts_raw)
      {
        if (p.depth_lidar_x >= lo && p.depth_lidar_x <= hi)
        {
          bbox_pts.push_back(p.lidar);
        }
      }
      if (bbox_pts.size() < min_bbox_pts)
      {
        return false;
      }

      xs.clear();
      ys.clear();
      zs.clear();
      xs.reserve(bbox_pts.size());
      ys.reserve(bbox_pts.size());
      zs.reserve(bbox_pts.size());
      for (const auto &p : bbox_pts)
      {
        xs.push_back(p[0]);
        ys.push_back(p[1]);
        zs.push_back(p[2]);
      }
      const float mx = MedianInPlace(xs);
      const float my = MedianInPlace(ys);
      const float mz = MedianInPlace(zs);

      const auto mad = [&](const std::vector<cv::Vec3f> &pts, int axis,
                           float med) -> float
      {
        absdev.clear();
        absdev.reserve(pts.size());
        for (const auto &p : pts)
        {
          absdev.push_back(std::fabs(p[axis] - med));
        }
        return MedianInPlace(absdev);
      };
      const float mad_x = mad(bbox_pts, 0, mx);
      const float mad_y = mad(bbox_pts, 1, my);
      const float mad_z = mad(bbox_pts, 2, mz);
      const float k = static_cast<float>(distance_cfg_.outlier_mad_k);
      const float min_abs = static_cast<float>(distance_cfg_.outlier_min_abs_m);
      const float thr_x = std::max(min_abs, k * mad_x);
      const float thr_y = std::max(min_abs, k * mad_y);
      const float thr_z = std::max(min_abs, k * mad_z);
      bbox_pts.erase(
          std::remove_if(bbox_pts.begin(), bbox_pts.end(),
                         [&](const cv::Vec3f &p)
                         {
                           return (std::fabs(p[0] - mx) > thr_x) ||
                                  (std::fabs(p[1] - my) > thr_y) ||
                                  (std::fabs(p[2] - mz) > thr_z);
                         }),
          bbox_pts.end());
      if (bbox_pts.size() < min_bbox_pts)
      {
        return false;
      }

      cv::Vec3f center{};
      cv::Vec3f size{};
      float yaw = 0.0f;
      std::array<cv::Vec3f, 8> corners{};
      if (!ComputeOrientedBboxZ(bbox_pts, center, size, yaw, corners))
      {
        return false;
      }

      out = Bbox3DResult{};
      out.valid = true;
      out.source_box = box;
      out.center_lidar = center;
      out.size_lidar = size;
      out.yaw_rad = yaw;
      out.corners_lidar = corners;
      project_bbox3d_corners(out);
      return true;
    };

    // Only attempt to fit 3D bounding boxes for certain classes (e.g., vehicles and pedestrians).
    // const std::unordered_set<int> enabled_labels = {0, 11};
    std::vector<Bbox3DResult> fitted_bbox3d;
    fitted_bbox3d.reserve(boxes.size());
    for (const auto &box : boxes)
    {
      // if (!enabled_labels.count(box.label))
      // {
      //   continue;
      // }

      Bbox3DResult bbox3d;
      if (try_fit_bbox3d(box, bbox3d))
      {
        fitted_bbox3d.push_back(std::move(bbox3d));
      }
    }

    std::vector<Bbox3DResult> cached_bbox3d;
    rclcpp::Time cached_stamp{0, 0, RCL_ROS_TIME};
    const bool allow_hold = bbox3d_cfg_.hold_ms > 0.0;
    if (allow_hold)
    {
      std::lock_guard<std::mutex> lk(bucket_bbox3d_cache_mutex_);
      cached_bbox3d = bucket_bbox3d_cache_;
      cached_stamp = bucket_bbox3d_cache_stamp_;
    }
    const bool cache_is_fresh =
        allow_hold && cached_stamp.nanoseconds() > 0 &&
        (static_cast<double>(std::llabs(stamp.nanoseconds() -
                                        cached_stamp.nanoseconds())) /
         1000000.0) <= bbox3d_cfg_.hold_ms;

    std::vector<uint8_t> fitted_used(fitted_bbox3d.size(), 0);
    std::vector<uint8_t> cache_used(cached_bbox3d.size(), 0);
    const auto find_best_match =
        [&](const Box &target, const std::vector<Bbox3DResult> &cands,
            std::vector<uint8_t> &used, float min_iou) -> int
    {
      int best_idx = -1;
      float best_iou = min_iou;
      for (size_t i = 0; i < cands.size(); ++i)
      {
        if (used[i] || !cands[i].valid)
        {
          continue;
        }
        if (cands[i].source_box.label != target.label)
        {
          continue;
        }
        const float iou = IoU2D(cands[i].source_box, target);
        if (iou >= best_iou)
        {
          best_iou = iou;
          best_idx = static_cast<int>(i);
        }
      }
      return best_idx;
    };

    bundle.bbox3d_objects.clear();
    bundle.bbox3d_objects.reserve(boxes.size());
    for (const auto &box : boxes)
    {
      const int fit_idx = find_best_match(box, fitted_bbox3d, fitted_used, 0.8f);
      if (fit_idx >= 0)
      {
        fitted_used[static_cast<size_t>(fit_idx)] = 1;
        bundle.bbox3d_objects.push_back(fitted_bbox3d[static_cast<size_t>(fit_idx)]);
        continue;
      }
      if (!cache_is_fresh)
      {
        continue;
      }
      const int cache_idx =
          find_best_match(box, cached_bbox3d, cache_used, 0.15f);
      if (cache_idx >= 0)
      {
        cache_used[static_cast<size_t>(cache_idx)] = 1;
        auto held = cached_bbox3d[static_cast<size_t>(cache_idx)];
        held.source_box = box;
        bundle.bbox3d_objects.push_back(std::move(held));
      }
    }

    if (!fitted_bbox3d.empty())
    {
      std::lock_guard<std::mutex> lk(bucket_bbox3d_cache_mutex_);
      bucket_bbox3d_cache_ = fitted_bbox3d;
      bucket_bbox3d_cache_stamp_ = stamp;
    }
  }

  bundle.bucket_bbox3d =
      bundle.bbox3d_objects.empty() ? Bbox3DResult{} : bundle.bbox3d_objects.front();

  return bundle;
}

void CameraCombinedWithLidarPipelineManager::submitDistanceRequest(
    const rclcpp::Time &stamp, const std::vector<Box> &boxes)
{
  std::lock_guard<std::mutex> lk(distance_req_mutex_);
  distance_req_.stamp = stamp;
  distance_req_.boxes = boxes;
  distance_req_.pending = true;
  distance_req_cond_.notify_one();
}

bool CameraCombinedWithLidarPipelineManager::tryGetDistanceCache(
    const rclcpp::Time &stamp, DistanceBundle &out_bundle)
{
  std::lock_guard<std::mutex> lk(distance_cache_mutex_);
  if (distance_cache_stamp_.nanoseconds() <= 0)
  {
    return false;
  }
  const int64_t dt_ns =
      stamp.nanoseconds() - distance_cache_stamp_.nanoseconds();
  const double dt_ms = static_cast<double>(std::llabs(dt_ns)) / 1000000.0;
  if (dt_ms > 500.0)
  {
    return false;
  }
  out_bundle = distance_cache_;
  return true;
}

void CameraCombinedWithLidarPipelineManager::distanceWorker()
{
  while (running_)
  {
    DistanceRequest req;
    {
      std::unique_lock<std::mutex> lk(distance_req_mutex_);
      distance_req_cond_.wait_for(lk, std::chrono::milliseconds(10), [&]()
                                  { return !running_ || distance_req_.pending; });
      if (!running_)
      {
        break;
      }
      if (!distance_req_.pending)
      {
        continue;
      }
      req = distance_req_;
      distance_req_.pending = false;
    }

    auto bundle = computeDistanceBundle(req.stamp, req);
    publishDistanceBundle(req.stamp, bundle);

    if (use_nitros_sub_ && bbox3d_cfg_.enable && bucket_bbox3d_marker_pub_)
    {
      visualization_msgs::msg::MarkerArray markers;

      std::string frame_id;
      {
        std::lock_guard<std::mutex> lk(lidar_cloud_history_mutex_);
        frame_id = last_lidar_frame_id_;
      }
      if (frame_id.empty())
      {
        frame_id = "lidar";
      }

      auto make_header = [&]()
      {
        std_msgs::msg::Header h;
        h.stamp = req.stamp;
        h.frame_id = frame_id;
        return h;
      };

      visualization_msgs::msg::Marker clear_all;
      clear_all.header = make_header();
      clear_all.action = visualization_msgs::msg::Marker::DELETEALL;
      markers.markers.push_back(clear_all);

      for (size_t obj_idx = 0; obj_idx < bundle.bbox3d_objects.size(); ++obj_idx)
      {
        const auto &b = bundle.bbox3d_objects[obj_idx];
        if (!b.valid)
        {
          continue;
        }
        const int base_id = static_cast<int>(obj_idx) * 2;

        visualization_msgs::msg::Marker cube;
        cube.header = make_header();
        cube.ns = "bucket_bbox3d";
        cube.id = base_id;
        cube.type = visualization_msgs::msg::Marker::CUBE;
        cube.action = visualization_msgs::msg::Marker::ADD;
        cube.pose.position.x = b.center_lidar[0];
        cube.pose.position.y = b.center_lidar[1];
        cube.pose.position.z = b.center_lidar[2];
        const double half = 0.5 * static_cast<double>(b.yaw_rad);
        cube.pose.orientation.x = 0.0;
        cube.pose.orientation.y = 0.0;
        cube.pose.orientation.z = std::sin(half);
        cube.pose.orientation.w = std::cos(half);
        cube.scale.x = std::max(1e-3f, b.size_lidar[0]);
        cube.scale.y = std::max(1e-3f, b.size_lidar[1]);
        cube.scale.z = std::max(1e-3f, b.size_lidar[2]);
        cube.color.r = 0.1f;
        cube.color.g = 1.0f;
        cube.color.b = 0.1f;
        cube.color.a = 0.25f;
        cube.lifetime.sec = 0;
        cube.lifetime.nanosec = 200000000;
        markers.markers.push_back(cube);

        visualization_msgs::msg::Marker edges;
        edges.header = make_header();
        edges.ns = "bucket_bbox3d";
        edges.id = base_id + 1;
        edges.type = visualization_msgs::msg::Marker::LINE_LIST;
        edges.action = visualization_msgs::msg::Marker::ADD;
        edges.scale.x = 0.06;
        edges.color.r = 0.1f;
        edges.color.g = 1.0f;
        edges.color.b = 0.1f;
        edges.color.a = 0.9f;
        edges.lifetime.sec = 0;
        edges.lifetime.nanosec = 200000000;

        auto push_edge = [&](int i, int j)
        {
          geometry_msgs::msg::Point p0;
          p0.x = b.corners_lidar[static_cast<size_t>(i)][0];
          p0.y = b.corners_lidar[static_cast<size_t>(i)][1];
          p0.z = b.corners_lidar[static_cast<size_t>(i)][2];
          geometry_msgs::msg::Point p1;
          p1.x = b.corners_lidar[static_cast<size_t>(j)][0];
          p1.y = b.corners_lidar[static_cast<size_t>(j)][1];
          p1.z = b.corners_lidar[static_cast<size_t>(j)][2];
          edges.points.push_back(p0);
          edges.points.push_back(p1);
        };
        for (const auto &edge : kBboxEdges)
        {
          push_edge(edge.first, edge.second);
        }
        markers.markers.push_back(edges);
      }

      bucket_bbox3d_marker_pub_->publish(markers);
    }
    {
      std::lock_guard<std::mutex> lk(distance_cache_mutex_);
      distance_cache_ = std::move(bundle);
      distance_cache_stamp_ = req.stamp;
    }
  }
}

void CameraCombinedWithLidarPipelineManager::computeAndPublishDistances(
    const rclcpp::Time &stamp, const std::vector<Box> &boxes)
{
  submitDistanceRequest(stamp, boxes);
}

void CameraCombinedWithLidarPipelineManager::publishDistanceBundle(
    const rclcpp::Time &stamp, const DistanceBundle &bundle)
{
  if (!detection_distance_pub_)
  {
    return;
  }

  nlohmann::json out;
  const int cam_id =
      (input_config_.camera_ids.empty() ||
       static_cast<size_t>(input_config_.idx) >=
           input_config_.camera_ids.size())
          ? 0
          : input_config_.camera_ids[static_cast<size_t>(input_config_.idx)];
  out["camera_id"] = cam_id;
  out["stamp_ns"] = stamp.nanoseconds();
  out["calib_valid"] = bundle.calib_valid;
  out["distance_frame"] = "lidar";
  if (bundle.cloud_valid)
  {
    out["cloud_stamp_ns"] = bundle.cloud_stamp_ns;
    out["cloud_age_ms"] = bundle.cloud_age_ms;
    out["cloud_points"] = bundle.cloud_points;
  }
  else
  {
    out["cloud_stamp_ns"] = nullptr;
    out["cloud_age_ms"] = nullptr;
    out["cloud_points"] = 0;
  }

  auto arr = nlohmann::json::array();

  for (const auto &m : bundle.objects)
  {
    nlohmann::json item;
    item["label"] = m.box.label;
    item["score"] = m.box.confidence;
    item["box"] = {m.box.left, m.box.top, m.box.right, m.box.bottom};
    item["center_px"] = {m.center_u, m.center_v};
    item["center_radius_px"] = m.center_radius_px;
    item["points_in_box"] = m.points_in_box;
    item["points_in_center"] = m.points_in_center;
    item["near_k"] = m.near_k;
    item["roi"] = (m.roi == 1   ? "tooth_band"
                   : m.roi == 2 ? "box"
                   : m.roi == 3 ? "hold"
                                : "none");
    if (m.has_distance)
    {
      item["range_m"] = m.range_m;
      item["depth_m"] = m.depth_m;
      item["height_m"] = m.height_m;
    }
    else
    {
      item["range_m"] = nullptr;
      item["depth_m"] = nullptr;
      item["height_m"] = nullptr;
    }

    arr.push_back(std::move(item));
  }

  out["objects"] = std::move(arr);

  auto bbox3d_arr = nlohmann::json::array();
  for (const auto &b : bundle.bbox3d_objects)
  {
    nlohmann::json bb;
    bb["label"] = b.source_box.label;
    bb["score"] = b.source_box.confidence;
    bb["box"] = {b.source_box.left, b.source_box.top, b.source_box.right,
                 b.source_box.bottom};
    bb["center_lidar"] = {b.center_lidar[0], b.center_lidar[1], b.center_lidar[2]};
    bb["size_lidar"] = {b.size_lidar[0], b.size_lidar[1], b.size_lidar[2]};
    bb["yaw_rad"] = b.yaw_rad;
    bbox3d_arr.push_back(std::move(bb));
  }
  out["bbox3d_objects"] = std::move(bbox3d_arr);

  if (bundle.bucket_bbox3d.valid)
  {
    const auto &b = bundle.bucket_bbox3d;
    nlohmann::json bb;
    bb["label"] = b.source_box.label;
    bb["score"] = b.source_box.confidence;
    bb["box"] = {b.source_box.left, b.source_box.top, b.source_box.right,
                 b.source_box.bottom};
    bb["center_lidar"] = {b.center_lidar[0], b.center_lidar[1], b.center_lidar[2]};
    bb["size_lidar"] = {b.size_lidar[0], b.size_lidar[1], b.size_lidar[2]};
    bb["yaw_rad"] = b.yaw_rad;
    out["bucket_bbox3d"] = std::move(bb);
  }
  else
  {
    out["bucket_bbox3d"] = nullptr;
  }

  std_msgs::msg::String msg;
  msg.data = out.dump();
  detection_distance_pub_->publish(msg);
}
