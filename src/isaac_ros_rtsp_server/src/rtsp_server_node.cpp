// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_rtsp_server/rtsp_server_node.hpp"

#include <algorithm>
#include <cctype>
#include <functional>

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_rtsp_server/nitros_compressed_image_view.hpp"

namespace isaac_ros_rtsp_server {
namespace {
std::string toLower(const std::string &value) {
  std::string out = value;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

size_t startCodeLength(const std::vector<uint8_t> &data, size_t index) {
  if (index + 3 <= data.size() && data[index] == 0 && data[index + 1] == 0 &&
      data[index + 2] == 1) {
    return 3;
  }
  if (index + 4 <= data.size() && data[index] == 0 && data[index + 1] == 0 &&
      data[index + 2] == 0 && data[index + 3] == 1) {
    return 4;
  }
  return 0;
}

bool hasAnnexBStartCode(const std::vector<uint8_t> &data) {
  return startCodeLength(data, 0) > 0;
}

bool forEachAnnexBNalu(
    const std::vector<uint8_t> &data,
    const std::function<void(const uint8_t *, size_t)> &callback) {
  if (!callback) {
    return false;
  }

  size_t start = data.size();
  size_t index = 0;
  while (index < data.size()) {
    const size_t sc_len = startCodeLength(data, index);
    if (sc_len > 0) {
      if (start < data.size() && index > start) {
        callback(data.data() + start, index - start);
      }
      start = index;
      index += sc_len;
      continue;
    }
    ++index;
  }

  if (start < data.size()) {
    callback(data.data() + start, data.size() - start);
    return true;
  }
  return false;
}
} // namespace

RtspServerNode::RtspServerNode(const rclcpp::NodeOptions &options)
    : Node("isaac_ros_rtsp_server", options) {
  config_ = std::make_shared<rtspcomponent::RtspServerConfig>();
  std::string video_type{"h264"};

  declare_parameter("stream_name", config_->stream_name);
  declare_parameter("port", config_->port_);
  declare_parameter("video_type", video_type);
  declare_parameter("topic_name", std::string("image_compressed"));
  declare_parameter("use_nitros", false);
  declare_parameter("nitros_format", std::string("nitros_compressed_image"));
  declare_parameter("assume_annexb", true);
  declare_parameter("dump_frame", 0);
  declare_parameter("max_queue_frames",
                    static_cast<int>(config_->max_queue_frames));
  declare_parameter("rtsp_tx_pub_period_ms", rtsp_tx_pub_period_ms_);

  get_parameter("stream_name", config_->stream_name);
  get_parameter("port", config_->port_);
  get_parameter("video_type", video_type);
  get_parameter("topic_name", topic_name_);
  get_parameter("use_nitros", use_nitros_);
  get_parameter("nitros_format", nitros_format_);
  get_parameter("assume_annexb", assume_annexb_);
  get_parameter("dump_frame", dump_frame_);
  get_parameter("rtsp_tx_pub_period_ms", rtsp_tx_pub_period_ms_);
  int max_queue_frames = static_cast<int>(config_->max_queue_frames);
  get_parameter("max_queue_frames", max_queue_frames);
  if (max_queue_frames < 1) {
    max_queue_frames = 1;
  }
  config_->max_queue_frames = static_cast<uint32_t>(max_queue_frames);

  video_type_ = toLower(video_type);
  if (video_type_ == "h264") {
    config_->video_type_ = rtspcomponent::VideoType::H264;
  } else if (video_type_ == "h265") {
    config_->video_type_ = rtspcomponent::VideoType::H265;
  } else {
    RCLCPP_ERROR(get_logger(),
                 "Unsupported video_type: %s (expected h264/h265)",
                 video_type.c_str());
    rclcpp::shutdown();
    return;
  }

  server_ = std::make_shared<rtspcomponent::RtspServer>(config_);
  if (!server_ || server_->Init() != 0 || server_->Start() != 0) {
    RCLCPP_ERROR(get_logger(), "Failed to create RtspServer");
    rclcpp::shutdown();
    return;
  }

  rtsp_tx_age_pub_ = create_publisher<std_msgs::msg::Float64>(
      "rtsp_tx_age_ms", rclcpp::SensorDataQoS());
  rtsp_tx_metrics_pub_ = create_publisher<std_msgs::msg::UInt64MultiArray>(
      "rtsp_tx_metrics", rclcpp::SensorDataQoS());
  H264Or5FramedSource::SetTxCallback(
      config_->stream_name, [this](uint64_t stamp_ns, uint64_t tx_ns) {
        (void)tx_ns;
        const uint64_t tx_now_ns =
            static_cast<uint64_t>(this->now().nanoseconds());
        if (stamp_ns == 0 || tx_now_ns < stamp_ns) {
          return;
        }

        uint64_t rx_input_ns = 0;
        {
          std::lock_guard<std::mutex> lk(rtsp_input_mu_);
          auto it = rtsp_input_by_stamp_ns_.find(stamp_ns);
          if (it != rtsp_input_by_stamp_ns_.end()) {
            rx_input_ns = it->second;
            rtsp_input_by_stamp_ns_.erase(it);
          }
        }

        if (rtsp_tx_metrics_pub_) {
          std_msgs::msg::UInt64MultiArray metrics;
          metrics.data.resize(3);
          metrics.data[0] = stamp_ns;
          metrics.data[1] = tx_now_ns - stamp_ns;
          metrics.data[2] = (rx_input_ns != 0 && tx_now_ns >= rx_input_ns)
                                ? (tx_now_ns - rx_input_ns)
                                : 0;
          rtsp_tx_metrics_pub_->publish(metrics);
        }

        if (rtsp_tx_pub_period_ms_ > 0) {
          std::lock_guard<std::mutex> lk(rtsp_tx_pub_mu_);
          if (last_rtsp_tx_pub_ns_ != 0 &&
              tx_now_ns - last_rtsp_tx_pub_ns_ <
                  static_cast<uint64_t>(rtsp_tx_pub_period_ms_) * 1000000ULL) {
            return;
          }
          last_rtsp_tx_pub_ns_ = tx_now_ns;
        }

        if (rtsp_tx_age_pub_) {
          std_msgs::msg::Float64 age;
          age.data = static_cast<double>(tx_now_ns - stamp_ns) / 1000000.0;
          rtsp_tx_age_pub_->publish(age);
        }
      });

  if (use_nitros_) {
    using nvidia::isaac_ros::nitros::ManagedNitrosSubscriber;
    using nvidia::isaac_ros::nitros::NitrosCompressedImageView;

    nitros_sub_ =
        std::make_shared<ManagedNitrosSubscriber<NitrosCompressedImageView>>(
            this, topic_name_, nitros_format_,
            [this](const NitrosCompressedImageView &view) {
              this->onNitrosCompressedImage(view.GetMessage());
            },
            nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
            rclcpp::SensorDataQoS());
  } else {
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    subscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        topic_name_, rclcpp::SensorDataQoS(),
        std::bind(&RtspServerNode::onCompressedImage, this,
                  std::placeholders::_1),
        sub_opts);
  }

  if (dump_frame_ == 1) {
    dump_file_.open("dump_stream.264", std::ios::out | std::ios::binary);
    RCLCPP_WARN(get_logger(), "Dump stream to dump_stream.264");
  }
}

RtspServerNode::~RtspServerNode() {
  if (config_) {
    H264Or5FramedSource::ClearTxCallback(config_->stream_name);
  }
  if (dump_frame_ == 1 && dump_file_.is_open()) {
    dump_file_.close();
  }
  if (server_) {
    server_->Stop();
  }
}

void RtspServerNode::onCompressedImage(
    sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
  if (!msg) {
    return;
  }
  // const int64_t now_ns = this->now().nanoseconds();
  // const int64_t msg_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
  // if (msg_ns > 0 && now_ns >= msg_ns) {
  //   const double age_ms = static_cast<double>(now_ns - msg_ns) / 1000000.0;
  //   RCLCPP_INFO_THROTTLE(
  //       get_logger(), *get_clock(), 1000, "[E2E] stream=%s topic=%s age=%.3fms",
  //       config_->stream_name.c_str(), topic_name_.c_str(), age_ms);
  // }
  processCompressedImage(*msg);
}

void RtspServerNode::onNitrosCompressedImage(
    const nvidia::isaac_ros::nitros::NitrosCompressedImage &msg) {
  nvidia::isaac_ros::nitros::NitrosCompressedImageView view(msg);
  // {
  //   const int64_t now_ns = this->now().nanoseconds();
  //   const int64_t msg_ns =
  //       static_cast<int64_t>(view.GetTimestampSeconds()) * 1000000000LL +
  //       static_cast<int64_t>(view.GetTimestampNanoseconds());
  //   if (msg_ns > 0 && now_ns >= msg_ns) {
  //     const double age_ms = static_cast<double>(now_ns - msg_ns) / 1000000.0;
  //     RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
  //                          "[E2E] stream=%s topic=%s age=%.3fms",
  //                          config_->stream_name.c_str(), topic_name_.c_str(),
  //                          age_ms);
  //   }
  // }
  sensor_msgs::msg::CompressedImage ros_msg;
  rclcpp::TypeAdapter<
      nvidia::isaac_ros::nitros::NitrosCompressedImage,
      sensor_msgs::msg::CompressedImage>::convert_to_ros_message(msg, ros_msg);
  ros_msg.header.stamp.sec = view.GetTimestampSeconds();
  ros_msg.header.stamp.nanosec = view.GetTimestampNanoseconds();
  processCompressedImage(ros_msg);
}

void RtspServerNode::processCompressedImage(
    const sensor_msgs::msg::CompressedImage &msg) {
  if (msg.data.empty()) {
    return;
  }
  const uint64_t stamp_ns =
      static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL +
      static_cast<uint64_t>(msg.header.stamp.nanosec);
  if (stamp_ns != 0) {
    const uint64_t rx_input_ns =
        static_cast<uint64_t>(this->now().nanoseconds());
    std::lock_guard<std::mutex> lk(rtsp_input_mu_);
    rtsp_input_by_stamp_ns_[stamp_ns] = rx_input_ns;
    if (rtsp_input_by_stamp_ns_.size() > 2048) {
      rtsp_input_by_stamp_ns_.clear();
    }
  }

  const std::string format = toLower(msg.format);
  if (format.find("h264") == std::string::npos &&
      format.find("h265") == std::string::npos) {
    if (!warned_format_) {
      warned_format_ = true;
      RCLCPP_WARN(
          get_logger(),
          "Compressed format is '%s' (expected h264/h265); stream may fail",
          msg.format.c_str());
    }
  }

  const bool has_start_code = hasAnnexBStartCode(msg.data);
  if (assume_annexb_ && !has_start_code && !warned_no_startcode_) {
    warned_no_startcode_ = true;
    RCLCPP_WARN(get_logger(),
                "Input does not look like AnnexB (no start code). "
                "Set assume_annexb:=false if you need conversion.");
  }

  const int media_type = (video_type_ == "h265")
                             ? static_cast<int>(rtspcomponent::VideoType::H265)
                             : static_cast<int>(rtspcomponent::VideoType::H264);
  if (assume_annexb_ && has_start_code) {
    const bool split = forEachAnnexBNalu(
        msg.data,
        [this, media_type, stamp_ns](const uint8_t *nalu, size_t nalu_size) {
          if (nalu_size > 0) {
            server_->SendData(nalu, static_cast<int>(nalu_size), media_type,
                              stamp_ns);
          }
        });
    if (!split) {
      server_->SendData(msg.data.data(), static_cast<int>(msg.data.size()),
                        media_type, stamp_ns);
    }
  } else {
    server_->SendData(msg.data.data(), static_cast<int>(msg.data.size()),
                      media_type, stamp_ns);
  }

  if (dump_frame_ == 1 && dump_file_.is_open()) {
    dump_file_.write(reinterpret_cast<const char *>(msg.data.data()),
                     static_cast<std::streamsize>(msg.data.size()));
  }
}

} // namespace isaac_ros_rtsp_server

