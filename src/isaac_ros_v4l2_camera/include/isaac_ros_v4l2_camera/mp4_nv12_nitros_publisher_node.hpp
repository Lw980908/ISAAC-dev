// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_V4L2_CAMERA__MP4_NV12_NITROS_PUBLISHER_NODE_HPP_
#define ISAAC_ROS_V4L2_CAMERA__MP4_NV12_NITROS_PUBLISHER_NODE_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "camera_info_manager/camera_info_manager.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

namespace nvidia {
namespace isaac_ros {
namespace v4l2_camera {

// 读取 MP4 文件并发布 NITROS NV12（GPU buffer，按 GXF 期望布局）。
class Mp4Nv12NitrosPublisherNode : public rclcpp::Node {
public:
  explicit Mp4Nv12NitrosPublisherNode(
      const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~Mp4Nv12NitrosPublisherNode() override;

private:
  void Run();

  // Parameters
  std::string file_path_;
  bool loop_{true};
  int gpu_id_{0};
  int width_{0};
  int height_{0};
  double publish_fps_{0.0};  // 0 = use source fps (fallback 30)
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string camera_link_frame_name_;
  std::string optical_frame_name_;
  std::string camera_info_url_;

  std::atomic<bool> running_{false};
  std::thread worker_;

  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>>
      image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  camera_info_manager::CameraInfoManager camera_info_manager_;

  // 性能优化：复用 GPU buffer 和 CUDA stream
  void *gpu_buffer_{nullptr};  // 预分配的 GPU buffer，避免每帧 malloc
  size_t gpu_buffer_size_{0};   // GPU buffer 大小
  cudaStream_t cuda_stream_{nullptr};  // CUDA stream 用于异步操作
};

}  // namespace v4l2_camera
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_V4L2_CAMERA__MP4_NV12_NITROS_PUBLISHER_NODE_HPP_

