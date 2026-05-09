#include "isaac_ros_rtsp_server/latency_monitor_node.hpp"

#include <chrono>

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "isaac_ros_rtsp_server/nitros_compressed_image_view.hpp"
#include "std_msgs/msg/u_int64_multi_array.hpp"

#include "rclcpp_components/register_node_macro.hpp"

namespace isaac_ros_rtsp_server {

namespace {
int64_t NowSteadyNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint64_t ToStampNs(int32_t sec, uint32_t nanosec) {
  return static_cast<uint64_t>(sec) * 1000000000ULL +
         static_cast<uint64_t>(nanosec);
}
} // namespace

LatencyMonitorNode::LatencyMonitorNode(const rclcpp::NodeOptions &options)
    : Node("isaac_ros_latency_monitor", options) {
  std::vector<std::string> camera_namespaces = {"camera0", "camera1", "camera2",
                                                "camera3"};
  std::string image_topic = "image_raw";
  std::string processed_topic = "processed_image";
  std::string compressed_topic = "image_compressed";
  std::string image_format = "nitros_image_nv12";
  std::string compressed_format = "nitros_compressed_image";
  declare_parameter("camera_namespaces", camera_namespaces);
  declare_parameter("image_topic", image_topic);
  declare_parameter("processed_topic", processed_topic);
  declare_parameter("compressed_topic", compressed_topic);
  declare_parameter("image_format", image_format);
  declare_parameter("compressed_format", compressed_format);
  declare_parameter("log_period_ms", log_period_ms_);
  declare_parameter("max_pending", static_cast<int>(max_pending_));

  get_parameter("camera_namespaces", camera_namespaces);
  get_parameter("image_topic", image_topic);
  get_parameter("processed_topic", processed_topic);
  get_parameter("compressed_topic", compressed_topic);
  get_parameter("image_format", image_format);
  get_parameter("compressed_format", compressed_format);
  get_parameter("log_period_ms", log_period_ms_);
  int max_pending_i = static_cast<int>(max_pending_);
  get_parameter("max_pending", max_pending_i);
  if (max_pending_i < 10) {
    max_pending_i = 10;
  }
  max_pending_ = static_cast<size_t>(max_pending_i);

  using nvidia::isaac_ros::nitros::ManagedNitrosSubscriber;
  using nvidia::isaac_ros::nitros::NitrosCompressedImageView;
  using nvidia::isaac_ros::nitros::NitrosImageView;

  const rclcpp::QoS qos_profile = rclcpp::SensorDataQoS();

  subs_.reserve(camera_namespaces.size() * 3);
  for (const auto &ns : camera_namespaces) {
    const std::string image_full_topic = "/" + ns + "/" + image_topic;
    const std::string processed_full_topic = "/" + ns + "/" + processed_topic;
    const std::string compressed_full_topic = "/" + ns + "/" + compressed_topic;
    const std::string rtsp_tx_metrics_topic = "/" + ns + "/rtsp_tx_metrics";

    subs_.push_back(std::make_shared<ManagedNitrosSubscriber<NitrosImageView>>(
        this, image_full_topic, image_format,
        [this, ns](const NitrosImageView &view) {
          this->onImageStamp(ns, false, view.GetTimestampSeconds(),
                             view.GetTimestampNanoseconds());
        },
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, qos_profile));

    subs_.push_back(std::make_shared<ManagedNitrosSubscriber<NitrosImageView>>(
        this, processed_full_topic, image_format,
        [this, ns](const NitrosImageView &view) {
          this->onImageStamp(ns, true, view.GetTimestampSeconds(),
                             view.GetTimestampNanoseconds());
        },
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, qos_profile));

    subs_.push_back(
        std::make_shared<ManagedNitrosSubscriber<NitrosCompressedImageView>>(
            this, compressed_full_topic, compressed_format,
            [this, ns](const NitrosCompressedImageView &view) {
              this->onCompressedStamp(ns, view.GetTimestampSeconds(),
                                      view.GetTimestampNanoseconds());
            },
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, qos_profile));

    subs_.push_back(create_subscription<std_msgs::msg::UInt64MultiArray>(
        rtsp_tx_metrics_topic, qos_profile,
        [this, ns](const std_msgs::msg::UInt64MultiArray::ConstSharedPtr msg) {
          if (!msg) {
            return;
          }
          this->onRtspTxMetrics(ns, *msg);
        }));
  }
}

void LatencyMonitorNode::onImageStamp(const std::string &stream, bool processed,
                                      int32_t sec, uint32_t nanosec) {
  const uint64_t stamp_ns = ToStampNs(sec, nanosec);
  const int64_t rx_ns = NowSteadyNs();

  auto &state = streams_[stream];
  if (state.last_log_tp.time_since_epoch().count() == 0) {
    state.last_log_tp = std::chrono::steady_clock::now();
  }

  const bool inserted = (state.pending.find(stamp_ns) == state.pending.end());
  auto &entry = state.pending[stamp_ns];
  if (inserted) {
    state.order.push_back(stamp_ns);
    while (state.order.size() > max_pending_) {
      const uint64_t old = state.order.front();
      state.order.pop_front();
      state.pending.erase(old);
    }
  }

  if (processed) {
    entry.rx_processed_ns = rx_ns;
  } else {
    entry.rx_image_ns = rx_ns;
  }

  if (!(entry.rx_image_ns >= 0 && entry.rx_processed_ns >= 0 &&
        entry.rx_compressed_ns >= 0)) {
    return;
  }

  if (!entry.has_tx) {
    auto it_tx = state.tx_metrics_by_stamp.find(stamp_ns);
    if (it_tx != state.tx_metrics_by_stamp.end()) {
      entry.cam_to_tx_ms = it_tx->second.first;
      entry.enc_to_tx_ms = it_tx->second.second;
      entry.has_tx = true;
      state.tx_metrics_by_stamp.erase(it_tx);
    }
  }
  if (!entry.has_tx) {
    return;
  }

  state.latest_stamp_ns = stamp_ns;
  state.latest_cam_to_processed_ms =
      (entry.rx_processed_ns >= entry.rx_image_ns)
          ? (static_cast<double>(entry.rx_processed_ns - entry.rx_image_ns) /
             1000000.0)
          : -1.0;
  state.latest_processed_to_compressed_ms =
      (entry.rx_compressed_ns >= entry.rx_processed_ns)
          ? (static_cast<double>(entry.rx_compressed_ns -
                                 entry.rx_processed_ns) /
             1000000.0)
          : -1.0;
  state.latest_cam_to_compressed_ms =
      (entry.rx_compressed_ns >= entry.rx_image_ns)
          ? (static_cast<double>(entry.rx_compressed_ns - entry.rx_image_ns) /
             1000000.0)
          : -1.0;
  state.latest_cam_to_tx_ms = entry.cam_to_tx_ms;
  state.latest_enc_to_tx_ms = entry.enc_to_tx_ms;

  const int64_t now_ns = this->now().nanoseconds();
  if (now_ns >= static_cast<int64_t>(stamp_ns)) {
    state.latest_age_ms =
        static_cast<double>(now_ns - static_cast<int64_t>(stamp_ns)) /
        1000000.0;
  } else {
    state.latest_age_ms = -1.0;
  }

  state.pending.erase(stamp_ns);
  maybeLogStream(stream, state);
}

void LatencyMonitorNode::onCompressedStamp(const std::string &stream,
                                           int32_t sec, uint32_t nanosec) {
  const uint64_t stamp_ns = ToStampNs(sec, nanosec);
  const int64_t rx_ns = NowSteadyNs();

  auto &state = streams_[stream];
  if (state.last_log_tp.time_since_epoch().count() == 0) {
    state.last_log_tp = std::chrono::steady_clock::now();
  }

  const bool inserted = (state.pending.find(stamp_ns) == state.pending.end());
  auto &entry = state.pending[stamp_ns];
  if (inserted) {
    state.order.push_back(stamp_ns);
    while (state.order.size() > max_pending_) {
      const uint64_t old = state.order.front();
      state.order.pop_front();
      state.pending.erase(old);
    }
  }

  entry.rx_compressed_ns = rx_ns;

  if (!(entry.rx_image_ns >= 0 && entry.rx_processed_ns >= 0 &&
        entry.rx_compressed_ns >= 0)) {
    return;
  }

  if (!entry.has_tx) {
    auto it_tx = state.tx_metrics_by_stamp.find(stamp_ns);
    if (it_tx != state.tx_metrics_by_stamp.end()) {
      entry.cam_to_tx_ms = it_tx->second.first;
      entry.enc_to_tx_ms = it_tx->second.second;
      entry.has_tx = true;
      state.tx_metrics_by_stamp.erase(it_tx);
    }
  }
  if (!entry.has_tx) {
    return;
  }

  state.latest_stamp_ns = stamp_ns;
  state.latest_cam_to_processed_ms =
      (entry.rx_processed_ns >= entry.rx_image_ns)
          ? (static_cast<double>(entry.rx_processed_ns - entry.rx_image_ns) /
             1000000.0)
          : -1.0;
  state.latest_processed_to_compressed_ms =
      (entry.rx_compressed_ns >= entry.rx_processed_ns)
          ? (static_cast<double>(entry.rx_compressed_ns -
                                 entry.rx_processed_ns) /
             1000000.0)
          : -1.0;
  state.latest_cam_to_compressed_ms =
      (entry.rx_compressed_ns >= entry.rx_image_ns)
          ? (static_cast<double>(entry.rx_compressed_ns - entry.rx_image_ns) /
             1000000.0)
          : -1.0;
  state.latest_cam_to_tx_ms = entry.cam_to_tx_ms;
  state.latest_enc_to_tx_ms = entry.enc_to_tx_ms;

  const int64_t now_ns = this->now().nanoseconds();
  if (now_ns >= static_cast<int64_t>(stamp_ns)) {
    state.latest_age_ms =
        static_cast<double>(now_ns - static_cast<int64_t>(stamp_ns)) /
        1000000.0;
  } else {
    state.latest_age_ms = -1.0;
  }

  state.pending.erase(stamp_ns);
  maybeLogStream(stream, state);
}

void LatencyMonitorNode::onRtspTxMetrics(
    const std::string &stream, const std_msgs::msg::UInt64MultiArray &msg) {
  if (msg.data.size() < 3) {
    return;
  }
  const uint64_t stamp_ns = msg.data[0];
  const double cam_to_tx_ms = static_cast<double>(msg.data[1]) / 1000000.0;
  const double enc_to_tx_ms =
      (msg.data[2] != 0) ? (static_cast<double>(msg.data[2]) / 1000000.0)
                         : -1.0;

  auto &state = streams_[stream];
  auto it = state.pending.find(stamp_ns);
  if (it != state.pending.end()) {
    it->second.cam_to_tx_ms = cam_to_tx_ms;
    it->second.enc_to_tx_ms = enc_to_tx_ms;
    it->second.has_tx = true;
    const bool complete =
        (it->second.rx_image_ns >= 0 && it->second.rx_processed_ns >= 0 &&
         it->second.rx_compressed_ns >= 0);
    if (complete) {
      state.latest_stamp_ns = stamp_ns;
      state.latest_cam_to_processed_ms =
          (it->second.rx_processed_ns >= it->second.rx_image_ns)
              ? (static_cast<double>(it->second.rx_processed_ns -
                                     it->second.rx_image_ns) /
                 1000000.0)
              : -1.0;
      state.latest_processed_to_compressed_ms =
          (it->second.rx_compressed_ns >= it->second.rx_processed_ns)
              ? (static_cast<double>(it->second.rx_compressed_ns -
                                     it->second.rx_processed_ns) /
                 1000000.0)
              : -1.0;
      state.latest_cam_to_compressed_ms =
          (it->second.rx_compressed_ns >= it->second.rx_image_ns)
              ? (static_cast<double>(it->second.rx_compressed_ns -
                                     it->second.rx_image_ns) /
                 1000000.0)
              : -1.0;
      state.latest_cam_to_tx_ms = cam_to_tx_ms;
      state.latest_enc_to_tx_ms = enc_to_tx_ms;
      state.pending.erase(it);
      maybeLogStream(stream, state);
    }
    return;
  }

  state.tx_metrics_by_stamp[stamp_ns] = {cam_to_tx_ms, enc_to_tx_ms};
  if (state.tx_metrics_by_stamp.size() > max_pending_) {
    state.tx_metrics_by_stamp.clear();
  }
}

void LatencyMonitorNode::maybeLogStream(const std::string &stream,
                                        StreamState &state) {
  const auto now_tp = std::chrono::steady_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now_tp - state.last_log_tp)
                              .count();
  if (elapsed_ms < log_period_ms_) {
    return;
  }
  state.last_log_tp = now_tp;

  if (state.latest_stamp_ns == 0) {
    return;
  }

  double total_ms = -1.0;
  const int64_t now_ns = this->now().nanoseconds();
  if (now_ns >= static_cast<int64_t>(state.latest_stamp_ns)) {
    total_ms = static_cast<double>(
                   now_ns - static_cast<int64_t>(state.latest_stamp_ns)) /
               1000000.0;
  }

  RCLCPP_INFO(get_logger(),
              "[PIPE] stream=%s cam->proc=%.3fms proc->enc=%.3fms "
              "enc->tx=%.3fms total=%.3fms",
              stream.c_str(), state.latest_cam_to_processed_ms,
              state.latest_processed_to_compressed_ms,
              state.latest_enc_to_tx_ms, total_ms);
}

} // namespace isaac_ros_rtsp_server

RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_rtsp_server::LatencyMonitorNode)

