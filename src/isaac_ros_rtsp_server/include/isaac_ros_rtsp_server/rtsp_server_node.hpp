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

#ifndef ISAAC_ROS_RTSP_SERVER__RTSP_SERVER_NODE_HPP_
#define ISAAC_ROS_RTSP_SERVER__RTSP_SERVER_NODE_HPP_

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "isaac_ros_nitros_compressed_image_type/nitros_compressed_image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/u_int64_multi_array.hpp"

#include "isaac_ros_rtsp_server/rtsp_component.hpp"
#include "isaac_ros_rtsp_server/rtsp_server.hpp"

namespace isaac_ros_rtsp_server {

class RtspServerNode : public rclcpp::Node {
public:
  explicit RtspServerNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~RtspServerNode() override;

private:
  void onCompressedImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);
  void onNitrosCompressedImage(
      const nvidia::isaac_ros::nitros::NitrosCompressedImage &msg);
  void processCompressedImage(const sensor_msgs::msg::CompressedImage &msg);

  std::shared_ptr<rtspcomponent::RtspServerConfig> config_;
  std::shared_ptr<rtspcomponent::RtspServer> server_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
      subscription_;
  std::shared_ptr<void> nitros_sub_;

  std::string topic_name_;
  std::string video_type_;
  std::string nitros_format_;
  bool use_nitros_{false};
  bool assume_annexb_;
  int dump_frame_;
  std::fstream dump_file_;
  bool warned_no_startcode_{false};
  bool warned_format_{false};

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr rtsp_tx_age_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt64MultiArray>::SharedPtr
      rtsp_tx_metrics_pub_;
  std::mutex rtsp_tx_pub_mu_;
  uint64_t last_rtsp_tx_pub_ns_{0};
  int rtsp_tx_pub_period_ms_{1000};

  std::mutex rtsp_input_mu_;
  std::unordered_map<uint64_t, uint64_t> rtsp_input_by_stamp_ns_;
};

} // namespace isaac_ros_rtsp_server

#endif // ISAAC_ROS_RTSP_SERVER__RTSP_SERVER_NODE_HPP_
