#ifndef ISAAC_ROS_RTSP_SERVER__LATENCY_MONITOR_NODE_HPP_
#define ISAAC_ROS_RTSP_SERVER__LATENCY_MONITOR_NODE_HPP_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int64_multi_array.hpp"

namespace isaac_ros_rtsp_server {

class LatencyMonitorNode : public rclcpp::Node {
public:
  explicit LatencyMonitorNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  struct StageTimes {
    int64_t rx_image_ns{-1};
    int64_t rx_processed_ns{-1};
    int64_t rx_compressed_ns{-1};
    double cam_to_tx_ms{-1.0};
    double enc_to_tx_ms{-1.0};
    bool has_tx{false};
  };

  struct StreamState {
    std::unordered_map<uint64_t, StageTimes> pending;
    std::deque<uint64_t> order;
    std::unordered_map<uint64_t, std::pair<double, double>> tx_metrics_by_stamp;

    uint64_t latest_stamp_ns{0};
    double latest_cam_to_processed_ms{-1.0};
    double latest_processed_to_compressed_ms{-1.0};
    double latest_cam_to_compressed_ms{-1.0};
    double latest_age_ms{-1.0};
    double latest_cam_to_tx_ms{-1.0};
    double latest_enc_to_tx_ms{-1.0};

    std::chrono::steady_clock::time_point last_log_tp{};
  };

  void onImageStamp(const std::string &stream, bool processed, int32_t sec,
                    uint32_t nanosec);
  void onCompressedStamp(const std::string &stream, int32_t sec,
                         uint32_t nanosec);
  void onRtspTxMetrics(const std::string &stream,
                       const std_msgs::msg::UInt64MultiArray &msg);

  void maybeLogStream(const std::string &stream, StreamState &state);

  std::unordered_map<std::string, StreamState> streams_;
  std::vector<std::shared_ptr<void>> subs_;
  int log_period_ms_{1000};
  size_t max_pending_{600};
};

} // namespace isaac_ros_rtsp_server

#endif

