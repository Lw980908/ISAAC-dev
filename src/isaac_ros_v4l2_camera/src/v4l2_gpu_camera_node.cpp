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

#include "../include/isaac_ros_v4l2_camera/v4l2_gpu_camera_node.hpp"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <errno.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "std_msgs/msg/header.hpp"

#include "isaac_ros_v4l2_camera/nv12_format_convert.hpp"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wpedantic"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"
#pragma GCC diagnostic pop

// EGLImage 互操作头文件
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda_egl_interop.h>

namespace nvidia {
namespace isaac_ros {
namespace v4l2_camera {

struct V4l2GpuCameraNode::CudaBufferPool {
  explicit CudaBufferPool(size_t bytes) : size_bytes(bytes) {}
  CudaBufferPool(const CudaBufferPool &) = delete;
  CudaBufferPool &operator=(const CudaBufferPool &) = delete;

  ~CudaBufferPool() {
    std::lock_guard<std::mutex> lk(m);
    for (void *p : free_list) {
      if (p) {
        cudaFree(p);
      }
    }
    free_list.clear();
  }

  void warmup(size_t count) {
    std::lock_guard<std::mutex> lk(m);
    while (free_list.size() < count) {
      void *p = nullptr;
      if (cudaMalloc(&p, size_bytes) != cudaSuccess) {
        break;
      }
      free_list.push_back(p);
    }
  }

  void *acquire() {
    std::lock_guard<std::mutex> lk(m);
    if (!free_list.empty()) {
      void *p = free_list.back();
      free_list.pop_back();
      return p;
    }
    void *p = nullptr;
    if (cudaMalloc(&p, size_bytes) != cudaSuccess) {
      return nullptr;
    }
    return p;
  }

  void release(void *p) {
    if (!p) {
      return;
    }
    std::lock_guard<std::mutex> lk(m);
    free_list.push_back(p);
  }

  size_t size_bytes{0};
  std::mutex m;
  std::vector<void *> free_list;
};

namespace {

constexpr char OUTPUT_TOPIC_NAME_LEFT_IMAGE[] = "left/image_raw";
constexpr char OUTPUT_TOPIC_NAME_LEFT_CAMERAINFO[] = "left/camera_info";

constexpr int kDefaultWidth = 1920;
constexpr int kDefaultHeight = 1536;
constexpr int kDefaultFps = 30;
constexpr int kDefaultBufferCount = 2; // 2 buffers is enough for one frame
constexpr char kDefaultDevice[] = "/dev/video0";
constexpr int64_t kNsPerSec = 1000000000LL;

inline void ThrowErrno(const std::string &context) {
  std::stringstream error_msg;
  error_msg << context << ": " << strerror(errno);
  throw std::runtime_error(error_msg.str());
}

inline void CheckCuda(cudaError_t status, const std::string &context) {
  if (status != cudaSuccess) {
    std::stringstream error_msg;
    error_msg << context << ": " << cudaGetErrorString(status);
    throw std::runtime_error(error_msg.str());
  }
}

constexpr uint32_t AlignUp(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1U) / alignment * alignment;
}

const char *PtrTypeToString(cudaMemoryType type) {
  switch (type) {
  case cudaMemoryTypeHost:
    return "host";
  case cudaMemoryTypeDevice:
    return "device";
  case cudaMemoryTypeManaged:
    return "managed";
  default:
    return "unknown";
  }
}

void LogPointerType(rclcpp::Logger logger, const char *name, const void *ptr) {
  cudaPointerAttributes attr{};
#if CUDART_VERSION >= 10000
  cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
  if (status == cudaSuccess) {
    RCLCPP_INFO(logger, "%s ptr=%p type=%s", name, ptr,
                PtrTypeToString(attr.type));
  } else {
    RCLCPP_WARN(logger, "%s ptr=%p cudaPointerGetAttributes failed: %s", name,
                ptr, cudaGetErrorString(status));
  }
#else
  cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
  if (status == cudaSuccess) {
    RCLCPP_INFO(logger, "%s ptr=%p type=%s", name, ptr,
                PtrTypeToString(attr.memoryType));
  } else {
    RCLCPP_WARN(logger, "%s ptr=%p cudaPointerGetAttributes failed: %s", name,
                ptr, cudaGetErrorString(status));
  }
#endif
}

uint32_t ToV4L2PixelFormat(const std::string &pixel_format) {
  if (pixel_format == "YUYV") {
    return V4L2_PIX_FMT_YUYV;
  }
  throw std::runtime_error("Unsupported pixel_format: " + pixel_format);
}

int64_t ReadRtcpuOffsetNsFallback() {
#if defined(__aarch64__)
  uint64_t cycles = 0;
  uint64_t frequency = 0;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
  asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
  if (frequency == 0) {
    return -1;
  }

  timespec tp{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp) != 0) {
    return -1;
  }

  const auto tsc_ns = static_cast<uint64_t>(
      (static_cast<__uint128_t>(cycles) * kNsPerSec) / frequency);
  const auto raw_ns = static_cast<uint64_t>(tp.tv_sec) * kNsPerSec +
                      static_cast<uint64_t>(tp.tv_nsec);
  return static_cast<int64_t>((tsc_ns > raw_ns) ? (tsc_ns - raw_ns)
                                                : (raw_ns - tsc_ns));
#else
  return -1;
#endif
}

int64_t LoadRtcpuOffsetNs() {
  char buf[128] = {0};
  FILE *fp =
      fopen("/sys/devices/system/clocksource/clocksource0/offset_ns", "r");
  if (fp != nullptr) {
    if (fgets(buf, sizeof(buf), fp) != nullptr) {
      fclose(fp);
      return static_cast<int64_t>(std::atoll(buf));
    }
    fclose(fp);
  }

  return ReadRtcpuOffsetNsFallback();
}

timespec NormalizeTimespec(timespec ts) {
  while (ts.tv_nsec >= kNsPerSec) {
    ++ts.tv_sec;
    ts.tv_nsec -= kNsPerSec;
  }
  while (ts.tv_nsec < 0) {
    --ts.tv_sec;
    ts.tv_nsec += kNsPerSec;
  }
  return ts;
}

timespec RtcpuToRealtime(const timeval &rtcpu_time, int64_t offset_ns) {
  timespec real_sample{};
  timespec monotonic_sample{};
  timespec monotonic_time{};

  const int64_t ns = static_cast<int64_t>(rtcpu_time.tv_sec) * kNsPerSec +
                     static_cast<int64_t>(rtcpu_time.tv_usec) * 1000LL -
                     offset_ns;
  monotonic_time.tv_sec = ns / kNsPerSec;
  monotonic_time.tv_nsec = ns % kNsPerSec;
  monotonic_time = NormalizeTimespec(monotonic_time);

  clock_gettime(CLOCK_MONOTONIC_RAW, &monotonic_sample);
  clock_gettime(CLOCK_REALTIME, &real_sample);

  timespec real_time{};
  real_time.tv_sec =
      monotonic_time.tv_sec + (real_sample.tv_sec - monotonic_sample.tv_sec);
  real_time.tv_nsec = monotonic_time.tv_nsec +
                      (real_sample.tv_nsec - monotonic_sample.tv_nsec);
  return NormalizeTimespec(real_time);
}

const char *GetTimestampSourceDescription(bool has_v4l2_timestamp,
                                          bool has_rtcpu_offset) {
  if (has_v4l2_timestamp) {
    return has_rtcpu_offset ? "capture_time: v4l2 buf.timestamp -> rtcpu_to_realtime"
                            : "capture_time: raw v4l2 buf.timestamp";
  }
  return "fallback_time: node current time now()";
}

} // namespace

V4l2GpuCameraNode::V4l2GpuCameraNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("v4l2_gpu_camera_node", options),
      camera_info_manager_(this, this->get_name()) {
  device_ = declare_parameter<std::string>("device", kDefaultDevice);
  width_ = declare_parameter<int>("width", kDefaultWidth);
  height_ = declare_parameter<int>("height", kDefaultHeight);
  framerate_ = declare_parameter<int>("framerate", kDefaultFps);
  gpu_id_ = declare_parameter<int>("gpu_id", 0);
  pixel_format_ = declare_parameter<std::string>("pixel_format", "YUYV");
  output_encoding_ = declare_parameter<std::string>("output_encoding", "nv12");
  transform_compute_mode_ =
#if defined(__aarch64__)
      declare_parameter<std::string>("transform_compute_mode", "vic");
#else
      declare_parameter<std::string>("transform_compute_mode", "gpu");
#endif
  buffer_count_ = declare_parameter<int>("buffer_count", kDefaultBufferCount);
  image_topic_ = declare_parameter<std::string>("image_topic",
                                                OUTPUT_TOPIC_NAME_LEFT_IMAGE);
  camera_info_topic_ = declare_parameter<std::string>(
      "camera_info_topic", OUTPUT_TOPIC_NAME_LEFT_CAMERAINFO);
  camera_link_frame_name_ =
      declare_parameter<std::string>("camera_link_frame_name", "camera");
  optical_frame_name_ =
      declare_parameter<std::string>("optical_frame_name", "camera_optical");
  camera_info_url_ = declare_parameter<std::string>("camera_info_url", "");

  CheckCuda(cudaSetDevice(gpu_id_), "cudaSetDevice failed");

  for (auto &c : transform_compute_mode_) {
    c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
  }

  if (!camera_info_url_.empty()) {
    camera_info_manager_.loadCameraInfo(camera_info_url_);
    RCLCPP_INFO(get_logger(), "Loaded camera info from \"%s\"",
                camera_info_url_.c_str());
  }

  const std::string output_encoding_lower = [&]() {
    std::string s = output_encoding_;
    for (auto &c : s) {
      c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    }
    return s;
  }();

  if (output_encoding_lower == "nv12") {
    image_pub_ = std::make_shared<
        nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
            nvidia::isaac_ros::nitros::NitrosImage>>(
        this, image_topic_,
        nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name);
  } else if (output_encoding_lower == "bgr8") {
    image_pub_ = std::make_shared<
        nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
            nvidia::isaac_ros::nitros::NitrosImage>>(
        this, image_topic_,
        nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name);
  } else {
    throw std::runtime_error("Unsupported output_encoding: " +
                             output_encoding_);
  }
  output_encoding_ = output_encoding_lower;

  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::QoS(1));

  InitializeCamera();
  running_.store(true);
  capture_thread_ = std::thread(&V4l2GpuCameraNode::CaptureLoop, this);
}

V4l2GpuCameraNode::~V4l2GpuCameraNode() {
  running_.store(false);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  ShutdownCamera();
}

void V4l2GpuCameraNode::InitializeCamera() {
  fd_ = open(device_.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd_ < 0) {
    ThrowErrno("Failed to open V4L2 device");
  }

  rtcpu_offset_ns_ = LoadRtcpuOffsetNs();
  if (rtcpu_offset_ns_ >= 0) {
    RCLCPP_INFO(get_logger(), "Using RTCPU timestamp offset: %lld ns",
                static_cast<long long>(rtcpu_offset_ns_));
  } else {
    RCLCPP_WARN(get_logger(),
                "Failed to load RTCPU offset, will fall back to raw V4L2 "
                "timestamps when available");
  }

  v4l2_capability cap{};
  if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
    ThrowErrno("VIDIOC_QUERYCAP failed");
  }
  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    throw std::runtime_error("V4L2 device does not support video capture");
  }

  RCLCPP_INFO(
      get_logger(),
      "Timestamp source plan before capture: prefer V4L2 buffer timestamp; %s",
      (rtcpu_offset_ns_ >= 0)
          ? "convert capture timestamp to realtime using RTCPU offset"
          : "if offset unavailable, use raw V4L2 timestamp directly");

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = width_;
  fmt.fmt.pix.height = height_;
  fmt.fmt.pix.pixelformat = ToV4L2PixelFormat(pixel_format_);
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  const uint32_t min_bytesperline = static_cast<uint32_t>(width_) * 2U;
  const uint32_t desired_bytesperline = AlignUp(min_bytesperline, 256U);
  fmt.fmt.pix.bytesperline = desired_bytesperline;
  fmt.fmt.pix.sizeimage = desired_bytesperline * static_cast<uint32_t>(height_);
  if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
    ThrowErrno("VIDIOC_S_FMT failed");
  }
  sizeimage_ = fmt.fmt.pix.sizeimage;
  src_pitch_ = (fmt.fmt.pix.bytesperline != 0)
                   ? static_cast<int>(fmt.fmt.pix.bytesperline)
                   : width_ * 2;
  RCLCPP_INFO(get_logger(), "V4L2 format: pixelformat=%u bytesperline=%u",
              fmt.fmt.pix.pixelformat, fmt.fmt.pix.bytesperline);
  RCLCPP_INFO(
      get_logger(), "V4L2 format: sizeimage=%u field=%u width=%u height=%u",
      sizeimage_, fmt.fmt.pix.field, fmt.fmt.pix.width, fmt.fmt.pix.height);
  if (fmt.fmt.pix.bytesperline != desired_bytesperline) {
    RCLCPP_WARN(get_logger(),
                "V4L2 bytesperline (%u) != desired aligned bytesperline (%u)",
                fmt.fmt.pix.bytesperline, desired_bytesperline);
  }

  // 设置 NvBufSurfTransform 参数
  CheckCuda(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags failed");

  NvBufSurfTransformConfigParams config{};
  config.compute_mode = NvBufSurfTransformCompute_GPU;
  config.gpu_id = gpu_id_;
  config.cuda_stream = cuda_stream_;

  if (transform_compute_mode_ == "vic") {
    config.compute_mode = NvBufSurfTransformCompute_VIC;
    config.cuda_stream = nullptr;
  } else if (transform_compute_mode_ == "gpu") {
    config.compute_mode = NvBufSurfTransformCompute_GPU;
    config.cuda_stream = cuda_stream_;
  } else {
    throw std::runtime_error("Unsupported transform_compute_mode: " +
                             transform_compute_mode_);
  }

  int session_status = NvBufSurfTransformSetSessionParams(&config);
  if (session_status != NvBufSurfTransformError_Success &&
      config.compute_mode == NvBufSurfTransformCompute_VIC) {
    RCLCPP_WARN(get_logger(),
                "NvBufSurfTransformSetSessionParams(VIC) failed (%d), fallback "
                "to GPU",
                session_status);
    config.compute_mode = NvBufSurfTransformCompute_GPU;
    config.cuda_stream = cuda_stream_;
    session_status = NvBufSurfTransformSetSessionParams(&config);
  }
  if (session_status != NvBufSurfTransformError_Success) {
    std::stringstream error_msg;
    error_msg << "NvBufSurfTransformSetSessionParams failed: "
              << session_status;
    throw std::runtime_error(error_msg.str());
  }

  // 设置帧率
  v4l2_streamparm streamparm{};
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = framerate_;
  if (ioctl(fd_, VIDIOC_S_PARM, &streamparm) < 0) {
    ThrowErrno("VIDIOC_S_PARM failed");
  }

  // 使用 V4L2_MEMORY_DMABUF 模式申请缓冲区
  v4l2_requestbuffers req{};
  req.count = static_cast<uint32_t>(buffer_count_);
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_DMABUF; // 使用 DMA-BUF 模式
  if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
    ThrowErrno("VIDIOC_REQBUFS with DMABUF failed");
  }
  if (req.count == 0) {
    throw std::runtime_error("V4L2 driver returned zero buffers");
  }
  RCLCPP_INFO(get_logger(), "V4L2 DMABUF mode: requested %d buffers, got %u",
              buffer_count_, req.count);

  // 为每个缓冲区创建 NvBufSurface（YUYV 格式）
  dmabuf_buffers_.resize(req.count);
  for (uint32_t i = 0; i < req.count; ++i) {
    NvBufSurfaceAllocateParams src_params{};
    src_params.params.width = width_;
    src_params.params.height = height_;
    src_params.params.layout = NVBUF_LAYOUT_PITCH;
    src_params.params.memType = NVBUF_MEM_SURFACE_ARRAY;
    src_params.params.colorFormat = NVBUF_COLOR_FORMAT_YUYV;
    src_params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

    if (NvBufSurfaceAllocate(&dmabuf_buffers_[i].surface, 1, &src_params) !=
        0) {
      throw std::runtime_error("NvBufSurfaceAllocate for DMABUF failed");
    }
    dmabuf_buffers_[i].surface->numFilled = 1;

    const uint32_t surface_pitch =
        dmabuf_buffers_[i].surface->surfaceList[0].pitch;
    if (src_pitch_ > 0 && surface_pitch != static_cast<uint32_t>(src_pitch_)) {
      std::stringstream error_msg;
      error_msg
          << "V4L2 bytesperline (" << src_pitch_
          << ") must match NvBufSurface pitch (" << surface_pitch
          << ") for DMABUF zero-copy. Consider using aligned bytesperline "
             "or adjusting capture width to a 128-pixel multiple.";
      throw std::runtime_error(error_msg.str());
    }

    // 获取 DMA-BUF fd
    dmabuf_buffers_[i].dmabuf_fd =
        dmabuf_buffers_[i].surface->surfaceList[0].bufferDesc;

    RCLCPP_INFO(
        get_logger(),
        "Allocated DMABUF buffer[%u]: fd=%d width=%u height=%u pitch=%u", i,
        dmabuf_buffers_[i].dmabuf_fd,
        dmabuf_buffers_[i].surface->surfaceList[0].width,
        dmabuf_buffers_[i].surface->surfaceList[0].height,
        dmabuf_buffers_[i].surface->surfaceList[0].pitch);

    QueueBuffer(i);
  }

  // 创建 NV12 输出 surface
  // 注意：NvBufSurfTransform 只支持 NVBUF_MEM_SURFACE_ARRAY 类型
  // 需要通过 EGLImage 互操作获取 CUDA 可访问的指针
  NvBufSurfaceAllocateParams dst_params{};
  dst_params.params.width = width_;
  dst_params.params.height = height_;
  dst_params.params.layout = NVBUF_LAYOUT_PITCH;
  dst_params.params.memType = NVBUF_MEM_SURFACE_ARRAY;
  // nv12: 2-plane; bgr8: 使用 BGRA 作为中间 surface（1-plane）
  dst_params.params.colorFormat = (output_encoding_ == "bgr8")
                                      ? NVBUF_COLOR_FORMAT_BGRA
                                      : NVBUF_COLOR_FORMAT_NV12;
  dst_params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;
  if (NvBufSurfaceAllocate(&dst_surface_, 1, &dst_params) != 0) {
    throw std::runtime_error("NvBufSurfaceAllocate dst failed");
  }
  dst_surface_->numFilled = 1;

  const size_t pool_bytes = (output_encoding_ == "bgr8")
                                ? static_cast<size_t>(width_) * height_ * 3
                                : static_cast<size_t>(width_) * height_ * 2;
  output_pool_ = std::make_shared<CudaBufferPool>(pool_bytes);
  output_pool_->warmup(static_cast<size_t>(std::max(2, buffer_count_)) * 2);

  const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
    ThrowErrno("VIDIOC_STREAMON failed");
  }

  RCLCPP_INFO(get_logger(), "Camera initialized with DMA-BUF zero-copy mode");
}

void V4l2GpuCameraNode::ShutdownCamera() {
  if (fd_ >= 0) {
    const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
    close(fd_);
    fd_ = -1;
  }

  // 释放 DMA-BUF 缓冲区
  for (auto &buf : dmabuf_buffers_) {
    if (buf.surface) {
      NvBufSurfaceDestroy(buf.surface);
      buf.surface = nullptr;
      buf.dmabuf_fd = -1;
    }
  }
  dmabuf_buffers_.clear();

  if (dst_surface_) {
    NvBufSurfaceDestroy(dst_surface_);
    dst_surface_ = nullptr;
  }

  output_pool_.reset();

  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }
}

void V4l2GpuCameraNode::QueueBuffer(uint32_t index) {
  if (index >= dmabuf_buffers_.size()) {
    throw std::runtime_error("QueueBuffer: invalid buffer index");
  }

  v4l2_buffer buf{};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_DMABUF; // 使用 DMA-BUF 模式
  buf.index = index;
  buf.m.fd = dmabuf_buffers_[index].dmabuf_fd; // 传入 DMA-BUF fd
  buf.bytesused = 0;

  if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
    ThrowErrno("VIDIOC_QBUF failed");
  }
}

void *V4l2GpuCameraNode::ConvertDmaBufToOutputCuda(uint32_t buf_index,
                                                   int height, int field) {
  (void)field;

  if (buf_index >= dmabuf_buffers_.size()) {
    throw std::runtime_error("ConvertDmaBufToOutputCuda: invalid buffer index");
  }
  if (!output_pool_) {
    throw std::runtime_error("Output pool is not initialized");
  }

  int effective_height = height;
  if (output_encoding_ == "nv12" && (effective_height % 2 != 0)) {
    effective_height -= 1;
  }
  if (effective_height <= 0) {
    throw std::runtime_error("ConvertDmaBufToOutputCuda: invalid height");
  }

  NvBufSurface *src_surface = dmabuf_buffers_[buf_index].surface;
  if (src_surface == nullptr || dst_surface_ == nullptr) {
    throw std::runtime_error("NvBufSurface not initialized");
  }

  NvBufSurfTransformParams params{};
  params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  params.transform_filter = NvBufSurfTransformInter_Default;
  const int transform_status =
      NvBufSurfTransform(src_surface, dst_surface_, &params);
  if (transform_status != NvBufSurfTransformError_Success) {
    std::stringstream error_msg;
    error_msg << "NvBufSurfTransform failed: " << transform_status;
    throw std::runtime_error(error_msg.str());
  }

  const auto &plane_params = dst_surface_->surfaceList[0].planeParams;

  void *frame_cuda_buffer = nullptr;
  bool egl_mapped = false;
  cudaGraphicsResource_t cuda_resource = nullptr;

  auto cleanup = [&]() {
    if (cuda_resource != nullptr) {
      cudaGraphicsUnregisterResource(cuda_resource);
      cuda_resource = nullptr;
    }
    if (egl_mapped) {
      NvBufSurfaceUnMapEglImage(dst_surface_, 0);
      egl_mapped = false;
    }
  };

  try {
    if (NvBufSurfaceMapEglImage(dst_surface_, 0) != 0) {
      throw std::runtime_error("NvBufSurfaceMapEglImage failed");
    }
    egl_mapped = true;

    EGLImageKHR egl_image = dst_surface_->surfaceList[0].mappedAddr.eglImage;
    if (egl_image == EGL_NO_IMAGE_KHR) {
      throw std::runtime_error("EGLImage is invalid");
    }

    cudaError_t cuda_status = cudaGraphicsEGLRegisterImage(
        &cuda_resource, egl_image, cudaGraphicsRegisterFlagsReadOnly);
    if (cuda_status != cudaSuccess) {
      std::stringstream error_msg;
      error_msg << "cudaGraphicsEGLRegisterImage failed: "
                << cudaGetErrorString(cuda_status);
      throw std::runtime_error(error_msg.str());
    }

    cudaEglFrame egl_frame{};
    cuda_status =
        cudaGraphicsResourceGetMappedEglFrame(&egl_frame, cuda_resource, 0, 0);
    if (cuda_status != cudaSuccess) {
      std::stringstream error_msg;
      error_msg << "cudaGraphicsResourceGetMappedEglFrame failed: "
                << cudaGetErrorString(cuda_status);
      throw std::runtime_error(error_msg.str());
    }

    cudaError_t kernel_status = cudaSuccess;

    if (output_encoding_ == "nv12") {
      const int dst_planes = static_cast<int>(plane_params.num_planes);
      if (dst_planes < 2) {
        throw std::runtime_error("NvBufSurface dst has <2 planes for nv12");
      }

      const void *src_y = egl_frame.frame.pPitch[0].ptr;
      const void *src_uv = egl_frame.frame.pPitch[1].ptr;
      if (src_y == nullptr || src_uv == nullptr) {
        throw std::runtime_error("EGLFrame pPitch is null for nv12");
      }

      const int y_pitch = static_cast<int>(plane_params.pitch[0]);
      const int uv_pitch = static_cast<int>(plane_params.pitch[1]);
      const int dst_y_stride = width_;
      const int dst_uv_stride = width_ * 2;

      frame_cuda_buffer = output_pool_->acquire();
      if (!frame_cuda_buffer) {
        throw std::runtime_error("Output pool acquire failed (nv12)");
      }

      kernel_status = ConvertNV12ToGxfLayout(
          frame_cuda_buffer, src_y, src_uv, width_, effective_height, y_pitch,
          uv_pitch, dst_y_stride, dst_uv_stride, cuda_stream_);
    } else if (output_encoding_ == "bgr8") {
      const int dst_planes = static_cast<int>(plane_params.num_planes);
      if (dst_planes < 1) {
        throw std::runtime_error("NvBufSurface dst has <1 planes for bgra");
      }

      const void *src_bgra = egl_frame.frame.pPitch[0].ptr;
      if (src_bgra == nullptr) {
        throw std::runtime_error("EGLFrame pPitch[0] is null for bgra");
      }

      const int bgra_pitch = static_cast<int>(plane_params.pitch[0]);
      const int dst_bgr_stride = width_ * 3;

      frame_cuda_buffer = output_pool_->acquire();
      if (!frame_cuda_buffer) {
        throw std::runtime_error("Output pool acquire failed (bgr8)");
      }

      kernel_status = ConvertBGRA8ToBgr8(frame_cuda_buffer, src_bgra, width_,
                                         effective_height, bgra_pitch,
                                         dst_bgr_stride, cuda_stream_);
    } else {
      throw std::runtime_error("Unsupported output_encoding: " +
                               output_encoding_);
    }

    cleanup();

    if (kernel_status != cudaSuccess) {
      output_pool_->release(frame_cuda_buffer);
      std::stringstream error_msg;
      error_msg << "GPU format convert failed: "
                << cudaGetErrorString(kernel_status);
      throw std::runtime_error(error_msg.str());
    }

    CheckCuda(cudaStreamSynchronize(cuda_stream_),
              "cudaStreamSynchronize failed");
    return frame_cuda_buffer;
  } catch (...) {
    cleanup();
    if (frame_cuda_buffer) {
      output_pool_->release(frame_cuda_buffer);
    }
    throw;
  }
}

void V4l2GpuCameraNode::PublishFrame(void *cuda_ptr, int height,
                                     const rclcpp::Time &stamp) {
  std_msgs::msg::Header header;
  header.stamp = stamp;
  header.frame_id = optical_frame_name_;

  if (output_encoding_ == "nv12" && (height % 2 != 0)) {
    height -= 1;
  }

  auto message = nvidia::gxf::Entity::New(
      nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext());
  if (!message) {
    std::stringstream error_msg;
    error_msg << "[PublishFrame] Error initializing entity: "
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
  if (!output_timestamp) {
    std::stringstream error_msg;
    error_msg << "[PublishFrame] Failed to add timestamp: "
              << GxfResultStr(output_timestamp.error());
    throw std::runtime_error(error_msg.str().c_str());
  }
  output_timestamp.value()->acqtime =
      static_cast<uint64_t>(header.stamp.sec) * kNsPerSec +
      static_cast<uint64_t>(header.stamp.nanosec);

  auto gxf_image =
      message->add<nvidia::gxf::VideoBuffer>(header.frame_id.c_str());
  if (!gxf_image) {
    std::stringstream error_msg;
    error_msg << "[PublishFrame] Failed to add VideoBuffer: "
              << GxfResultStr(gxf_image.error());
    throw std::runtime_error(error_msg.str().c_str());
  }

  using GxfVideoFormat = nvidia::gxf::VideoFormat;
  constexpr auto surface_layout =
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  constexpr auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;

  const uint32_t out_w = static_cast<uint32_t>(width_);
  const uint32_t out_h = static_cast<uint32_t>(height);
  uint64_t size = 0;
  nvidia::gxf::VideoBufferInfo buffer_info{};

  if (output_encoding_ == "nv12") {
    if (out_w % 2 != 0 || out_h % 2 != 0) {
      throw std::runtime_error(
          "[PublishFrame] NV12 requires even width/height");
    }
    const int y_stride = width_;
    const int uv_stride = width_ * 2;
    nvidia::gxf::VideoFormatSize<GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER>
        format_size;
    std::array<nvidia::gxf::ColorPlane, 2> planes{
        nvidia::gxf::ColorPlane("Y", 1, static_cast<uint32_t>(y_stride)),
        nvidia::gxf::ColorPlane("UV", 2, static_cast<uint32_t>(uv_stride)),
    };
    size = format_size.size(out_w, out_h, planes, /*stride_align=*/false);
    std::vector<nvidia::gxf::ColorPlane> color_planes{planes.begin(),
                                                      planes.end()};
    buffer_info = nvidia::gxf::VideoBufferInfo{
        out_w, out_h, GxfVideoFormat::GXF_VIDEO_FORMAT_NV12_ER,
        std::move(color_planes), surface_layout};
  } else if (output_encoding_ == "bgr8") {
    const int stride = width_ * 3;
    nvidia::gxf::VideoFormatSize<GxfVideoFormat::GXF_VIDEO_FORMAT_BGR>
        format_size;
    std::array<nvidia::gxf::ColorPlane, 1> planes{
        nvidia::gxf::ColorPlane("BGR", 3, static_cast<uint32_t>(stride)),
    };
    size = format_size.size(out_w, out_h, planes, /*stride_align=*/false);
    std::vector<nvidia::gxf::ColorPlane> color_planes{planes.begin(),
                                                      planes.end()};
    buffer_info = nvidia::gxf::VideoBufferInfo{
        out_w, out_h, GxfVideoFormat::GXF_VIDEO_FORMAT_BGR,
        std::move(color_planes), surface_layout};
  } else {
    throw std::runtime_error("Unsupported output_encoding: " +
                             output_encoding_);
  }

  gxf_image.value()->wrapMemory(buffer_info, size, storage_type, cuda_ptr,
                                [pool = output_pool_](void *ptr) {
                                  if (pool) {
                                    pool->release(ptr);
                                    return nvidia::gxf::Success;
                                  }
                                  cudaFree(ptr);
                                  return nvidia::gxf::Success;
                                });

  image_pub_->publish(nitros_image);

  auto camera_info = camera_info_manager_.getCameraInfo();
  camera_info.header.stamp = stamp;
  camera_info.header.frame_id = camera_link_frame_name_;
  camera_info.width = width_;
  camera_info.height = height;
  camera_info_pub_->publish(camera_info);
}

void V4l2GpuCameraNode::CaptureLoop() {
  CheckCuda(cudaSetDevice(gpu_id_), "cudaSetDevice in capture thread failed");
  while (rclcpp::ok() && running_.load()) {
    pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;
    const int poll_result = poll(&pfd, 1, 1000);
    if (poll_result < 0) {
      if (errno == EINTR) {
        continue;
      }
      RCLCPP_ERROR(get_logger(), "poll failed: %s", strerror(errno));
      continue;
    }
    if (poll_result == 0) {
      continue;
    }

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
      if (errno == EINTR) {
        continue;
      }
      RCLCPP_ERROR(get_logger(), "VIDIOC_DQBUF failed: %s", strerror(errno));
      continue;
    }

    if (buf.index >= dmabuf_buffers_.size()) {
      RCLCPP_ERROR(get_logger(), "Received out-of-range buffer index %u",
                   buf.index);
      continue;
    }

    static std::atomic_bool logged_frame{false};
    static std::atomic_bool logged_timestamp_source{false};
    if (!logged_frame.exchange(true)) {
      RCLCPP_INFO(get_logger(),
                  "V4L2 DMABUF frame: bytesused=%u sizeimage=%u field=%u "
                  "sequence=%u fd=%d",
                  buf.bytesused, sizeimage_, buf.field, buf.sequence,
                  dmabuf_buffers_[buf.index].dmabuf_fd);
    }

    try {
      const size_t min_bytes = static_cast<size_t>(src_pitch_) * height_;
      int actual_height = height_;
      rclcpp::Time capture_stamp = now();
      if (buf.bytesused > 0 && buf.bytesused < min_bytes) {
        actual_height = static_cast<int>(buf.bytesused / src_pitch_);
        if (actual_height <= 0) {
          actual_height = height_;
        }
        RCLCPP_WARN(
            get_logger(),
            "V4L2 bytesused smaller than frame, bytesused=%u expected=%zu "
            "height=%d -> %d",
            buf.bytesused, min_bytes, height_, actual_height);
      }

      if (buf.timestamp.tv_sec != 0 || buf.timestamp.tv_usec != 0) {
        if (rtcpu_offset_ns_ >= 0) {
          const timespec capture_time =
              RtcpuToRealtime(buf.timestamp, rtcpu_offset_ns_);
          capture_stamp = rclcpp::Time(
              static_cast<int64_t>(capture_time.tv_sec) * kNsPerSec +
                  static_cast<int64_t>(capture_time.tv_nsec),
              RCL_SYSTEM_TIME);
        } else {
          capture_stamp = rclcpp::Time(
              static_cast<int64_t>(buf.timestamp.tv_sec) * kNsPerSec +
                  static_cast<int64_t>(buf.timestamp.tv_usec) * 1000LL,
              RCL_SYSTEM_TIME);
        }
      }

      if (!logged_timestamp_source.exchange(true)) {
        const bool has_v4l2_timestamp =
            (buf.timestamp.tv_sec != 0 || buf.timestamp.tv_usec != 0);
        const rclcpp::Time now_stamp = now();
        const int64_t delay_ns = now_stamp.nanoseconds() - capture_stamp.nanoseconds();
        const double delay_ms = static_cast<double>(delay_ns) / 1000000.0;
        RCLCPP_INFO(
            get_logger(),
            "Timestamp source on first frame: %s (buf.timestamp=%ld.%06ld, "
            "rtcpu_offset_ns=%lld, capture_vs_now=%.3f ms)",
            GetTimestampSourceDescription(has_v4l2_timestamp,
                                          rtcpu_offset_ns_ >= 0),
            static_cast<long>(buf.timestamp.tv_sec),
            static_cast<long>(buf.timestamp.tv_usec),
            static_cast<long long>(rtcpu_offset_ns_), delay_ms);
      }

      void *cuda_ptr = ConvertDmaBufToOutputCuda(buf.index, actual_height,
                                                 static_cast<int>(buf.field));
      PublishFrame(cuda_ptr, actual_height, capture_stamp);
    } catch (const std::exception &ex) {
      RCLCPP_ERROR(get_logger(), "Capture error: %s", ex.what());
    }

    QueueBuffer(buf.index);
  }
}

} // namespace v4l2_camera
} // namespace isaac_ros
} // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
    nvidia::isaac_ros::v4l2_camera::V4l2GpuCameraNode)
