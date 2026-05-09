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

#ifndef ISAAC_ROS_V4L2_CAMERA__V4L2_GPU_CAMERA_NODE_HPP_
#define ISAAC_ROS_V4L2_CAMERA__V4L2_GPU_CAMERA_NODE_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "camera_info_manager/camera_info_manager.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

struct NvBufSurface;

namespace nvidia {
namespace isaac_ros {
namespace v4l2_camera {

class V4l2GpuCameraNode : public rclcpp::Node {
public:
  explicit V4l2GpuCameraNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~V4l2GpuCameraNode();

private:
  struct CudaBufferPool;

  // DMA-BUF 缓冲区信息
  struct DmaBufInfo {
    NvBufSurface *surface{nullptr}; // YUYV 输入 surface
    int dmabuf_fd{-1};              // DMA-BUF 文件描述符
  };

  void InitializeCamera();
  void ShutdownCamera();
  void CaptureLoop();
  void QueueBuffer(uint32_t index);
  void PublishFrame(void *cuda_ptr, int height, const rclcpp::Time &stamp);

  // DMA-BUF 零拷贝转换：直接从 NvBufSurface 转换到输出编码（nv12/bgr8）
  void *ConvertDmaBufToOutputCuda(uint32_t buf_index, int height, int field);

  std::string device_;
  int width_{0};
  int height_{0};
  int framerate_{0};
  int gpu_id_{0};
  std::string pixel_format_;
  std::string output_encoding_;
  std::string transform_compute_mode_;
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string camera_link_frame_name_;
  std::string optical_frame_name_;
  std::string camera_info_url_;
  int buffer_count_{0};

  int fd_{-1};
  std::vector<DmaBufInfo> dmabuf_buffers_; // DMA-BUF 缓冲区列表
  std::atomic<bool> running_{false};
  std::thread capture_thread_;

  int src_pitch_{0};
  uint32_t sizeimage_{0};
  NvBufSurface *dst_surface_{nullptr}; // 输出 surface（NV12 或 BGRA）

  cudaStream_t cuda_stream_{nullptr};
  std::shared_ptr<CudaBufferPool> output_pool_;

  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>>
      image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  camera_info_manager::CameraInfoManager camera_info_manager_;
};

} // namespace v4l2_camera
} // namespace isaac_ros
} // namespace nvidia

#endif // ISAAC_ROS_V4L2_CAMERA__V4L2_GPU_CAMERA_NODE_HPP_

