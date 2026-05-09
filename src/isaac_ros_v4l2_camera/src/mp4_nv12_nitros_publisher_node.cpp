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

#include "isaac_ros_v4l2_camera/mp4_nv12_nitros_publisher_node.hpp"

#include <cuda_runtime.h>

#include <cctype>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "std_msgs/msg/header.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace nvidia {
namespace isaac_ros {
namespace v4l2_camera {
namespace {

constexpr char kDefaultImageTopic[] = "image_raw";
constexpr char kDefaultCameraInfoTopic[] = "camera_info";

inline void CheckCuda(cudaError_t status, const std::string &context) {
  if (status != cudaSuccess) {
    std::stringstream ss;
    ss << context << ": " << cudaGetErrorString(status);
    throw std::runtime_error(ss.str());
  }
}

class AvError : public std::runtime_error {
public:
  explicit AvError(const std::string &context, int err)
      : std::runtime_error(Build(context, err)) {}

private:
  static std::string Build(const std::string &context, int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, buf, sizeof(buf));
    std::stringstream ss;
    ss << context << ": " << buf << " (err=" << err << ")";
    return ss.str();
  }
};

double GuessSourceFps(const AVStream *st) {
  if (!st) {
    return 30.0;
  }
  if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0) {
    return av_q2d(st->avg_frame_rate);
  }
  if (st->r_frame_rate.num > 0 && st->r_frame_rate.den > 0) {
    return av_q2d(st->r_frame_rate);
  }
  return 30.0;
}

struct AvFormatContextDeleter {
  void operator()(AVFormatContext *ctx) const {
    if (!ctx) {
      return;
    }
    AVFormatContext *tmp = ctx;
    avformat_close_input(&tmp);
  }
};

struct AvCodecContextDeleter {
  void operator()(AVCodecContext *ctx) const {
    if (!ctx) {
      return;
    }
    AVCodecContext *tmp = ctx;
    avcodec_free_context(&tmp);
  }
};

struct AvPacketDeleter {
  void operator()(AVPacket *pkt) const {
    if (!pkt) {
      return;
    }
    AVPacket *tmp = pkt;
    av_packet_free(&tmp);
  }
};

struct AvFrameDeleter {
  void operator()(AVFrame *frame) const {
    if (!frame) {
      return;
    }
    AVFrame *tmp = frame;
    av_frame_free(&tmp);
  }
};

} // namespace

Mp4Nv12NitrosPublisherNode::Mp4Nv12NitrosPublisherNode(
    const rclcpp::NodeOptions &options)
    : rclcpp::Node("mp4_nv12_nitros_publisher_node", options),
      camera_info_manager_(this, this->get_name()) {
  file_path_ = declare_parameter<std::string>("file_path", "");
  loop_ = declare_parameter<bool>("loop", true);
  gpu_id_ = declare_parameter<int>("gpu_id", 0);
  width_ = declare_parameter<int>("width", 0);
  height_ = declare_parameter<int>("height", 0);
  publish_fps_ = declare_parameter<double>("publish_fps", 0.0);
  image_topic_ =
      declare_parameter<std::string>("image_topic", kDefaultImageTopic);
  camera_info_topic_ = declare_parameter<std::string>("camera_info_topic",
                                                      kDefaultCameraInfoTopic);
  camera_link_frame_name_ =
      declare_parameter<std::string>("camera_link_frame_name", "camera");
  optical_frame_name_ =
      declare_parameter<std::string>("optical_frame_name", "camera_optical");
  camera_info_url_ = declare_parameter<std::string>("camera_info_url", "");

  if (file_path_.empty()) {
    throw std::runtime_error(
        "Parameter 'file_path' must be set to an MP4 file.");
  }
  if (width_ <= 0 || height_ <= 0) {
    // 为了让下游（感知/编码）在 launch/yaml 里能固定分辨率，强制要求显式配置
    throw std::runtime_error(
        "Parameters 'width' and 'height' must be set (>0).");
  }

  if (!camera_info_url_.empty()) {
    camera_info_manager_.loadCameraInfo(camera_info_url_);
    RCLCPP_INFO(get_logger(), "Loaded camera info from \"%s\"",
                camera_info_url_.c_str());
  }

  CheckCuda(cudaSetDevice(gpu_id_), "cudaSetDevice failed");

  // 创建 CUDA stream 用于异步操作
  CheckCuda(cudaStreamCreate(&cuda_stream_), "cudaStreamCreate failed");

  image_pub_ =
      std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
          this, image_topic_,
          nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name);

  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::QoS(1));

  running_.store(true);
  worker_ = std::thread(&Mp4Nv12NitrosPublisherNode::Run, this);
}

Mp4Nv12NitrosPublisherNode::~Mp4Nv12NitrosPublisherNode() {
  running_.store(false);
  if (worker_.joinable()) {
    worker_.join();
  }

  // 清理 GPU 资源
  if (cuda_stream_) {
    cudaStreamSynchronize(cuda_stream_);
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }
  if (gpu_buffer_) {
    cudaFree(gpu_buffer_);
    gpu_buffer_ = nullptr;
    gpu_buffer_size_ = 0;
  }
}

void Mp4Nv12NitrosPublisherNode::Run() {
  try {
    CheckCuda(cudaSetDevice(gpu_id_), "cudaSetDevice in worker thread failed");

    AVFormatContext *fmt = nullptr;
    int ret = avformat_open_input(&fmt, file_path_.c_str(), nullptr, nullptr);
    if (ret < 0) {
      throw AvError("avformat_open_input failed", ret);
    }

    std::unique_ptr<AVFormatContext, AvFormatContextDeleter> fmt_guard(fmt);

    ret = avformat_find_stream_info(fmt, nullptr);
    if (ret < 0) {
      throw AvError("avformat_find_stream_info failed", ret);
    }

    int video_stream_index = -1;
    for (unsigned int i = 0; i < fmt->nb_streams; ++i) {
      if (fmt->streams[i] && fmt->streams[i]->codecpar &&
          fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        video_stream_index = static_cast<int>(i);
        break;
      }
    }
    if (video_stream_index < 0) {
      throw std::runtime_error("No video stream found in MP4: " + file_path_);
    }

    AVStream *st = fmt->streams[video_stream_index];
    const AVCodec *decoder = avcodec_find_decoder(st->codecpar->codec_id);
    if (!decoder) {
      throw std::runtime_error("No decoder found for codec_id=" +
                               std::to_string(st->codecpar->codec_id));
    }

    AVCodecContext *dec_ctx = avcodec_alloc_context3(decoder);
    if (!dec_ctx) {
      throw std::runtime_error("avcodec_alloc_context3 failed");
    }
    std::unique_ptr<AVCodecContext, AvCodecContextDeleter> dec_guard(dec_ctx);

    ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
    if (ret < 0) {
      throw AvError("avcodec_parameters_to_context failed", ret);
    }

    ret = avcodec_open2(dec_ctx, decoder, nullptr);
    if (ret < 0) {
      throw AvError("avcodec_open2 failed", ret);
    }

    // 目标输出 NV12 的 swscale
    SwsContext *sws = nullptr;
    std::unique_ptr<SwsContext, decltype(&sws_freeContext)> sws_guard(
        nullptr, &sws_freeContext);

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
      throw std::runtime_error("av_packet_alloc failed");
    }
    std::unique_ptr<AVPacket, AvPacketDeleter> pkt_guard(pkt);

    AVFrame *frame = av_frame_alloc();
    if (!frame) {
      throw std::runtime_error("av_frame_alloc failed");
    }
    std::unique_ptr<AVFrame, AvFrameDeleter> frame_guard(frame);

    AVFrame *nv12 = av_frame_alloc();
    if (!nv12) {
      throw std::runtime_error("av_frame_alloc (nv12) failed");
    }
    std::unique_ptr<AVFrame, AvFrameDeleter> nv12_guard(nv12);

    nv12->format = AV_PIX_FMT_NV12;
    nv12->width = width_;
    nv12->height = height_;
    ret = av_frame_get_buffer(nv12, 32);
    if (ret < 0) {
      throw AvError("av_frame_get_buffer (nv12) failed", ret);
    }

    const double fps = (publish_fps_ > 0.0) ? publish_fps_ : GuessSourceFps(st);
    const auto frame_period =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / fps));
    auto next_deadline = std::chrono::steady_clock::now();

    RCLCPP_INFO(get_logger(),
                "MP4 NV12 publisher started. file=%s target=%dx%d loop=%s "
                "fps=%.3f gpu_id=%d",
                file_path_.c_str(), width_, height_, loop_ ? "true" : "false",
                fps, gpu_id_);

    while (rclcpp::ok() && running_.load()) {
      ret = av_read_frame(fmt, pkt);
      if (ret == AVERROR_EOF) {
        if (loop_) {
          avcodec_flush_buffers(dec_ctx);
          // seek to beginning (best effort)
          av_seek_frame(fmt, video_stream_index, 0, AVSEEK_FLAG_BACKWARD);
          continue;
        } else {
          break;
        }
      }
      if (ret < 0) {
        throw AvError("av_read_frame failed", ret);
      }

      if (pkt->stream_index != video_stream_index) {
        av_packet_unref(pkt);
        continue;
      }

      ret = avcodec_send_packet(dec_ctx, pkt);
      av_packet_unref(pkt);
      if (ret < 0) {
        throw AvError("avcodec_send_packet failed", ret);
      }

      while (rclcpp::ok() && running_.load()) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          break;
        }
        if (ret < 0) {
          throw AvError("avcodec_receive_frame failed", ret);
        }

        // init sws after first frame to know source format
        if (!sws) {
          // 性能优化：使用更快的算法（SWS_FAST_BILINEAR 比 SWS_BILINEAR 更快）
          // 如果质量要求高，可以改用 SWS_BILINEAR 或 SWS_LANCZOS
          sws = sws_getContext(frame->width, frame->height,
                               static_cast<AVPixelFormat>(frame->format),
                               width_, height_, AV_PIX_FMT_NV12,
                               SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
          if (!sws) {
            throw std::runtime_error("sws_getContext failed");
          }
          sws_guard.reset(sws);
          RCLCPP_INFO(
              get_logger(),
              "SwsContext initialized: %dx%d (%s) -> %dx%d (NV12)",
              frame->width, frame->height,
              av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)),
              width_, height_);
        }

        // 确保 nv12 buffer 可写
        ret = av_frame_make_writable(nv12);
        if (ret < 0) {
          throw AvError("av_frame_make_writable (nv12) failed", ret);
        }

        sws_scale(sws, frame->data, frame->linesize, 0, frame->height,
                  nv12->data, nv12->linesize);

        // ========== 复制到 GPU：按你现有 GXF 期望布局 ==========
        // y_size = width*height（dst_y_stride=width）
        // uv_size = (width*2)*(height/2)（dst_uv_stride=width*2，与你现有 nv12
        // 发布保持一致）
        const size_t y_size =
            static_cast<size_t>(width_) * static_cast<size_t>(height_);
        const size_t uv_size =
            static_cast<size_t>(width_) * 2U * static_cast<size_t>(height_ / 2);
        const size_t total_size = y_size + uv_size;

        // 性能优化：复用预分配的 GPU buffer，避免每帧 malloc
        if (!gpu_buffer_ || gpu_buffer_size_ < total_size) {
          if (gpu_buffer_) {
            cudaFree(gpu_buffer_);
          }
          CheckCuda(cudaMalloc(&gpu_buffer_, total_size),
                    "cudaMalloc NV12 GXF buffer failed");
          gpu_buffer_size_ = total_size;
        }

        uint8_t *dst_y = static_cast<uint8_t *>(gpu_buffer_);
        uint8_t *dst_uv = static_cast<uint8_t *>(gpu_buffer_) + y_size;

        // 性能优化：使用异步拷贝，不需要 memset（数据会被完全覆盖）
        // NV12(Y): copy width bytes per row into dst pitch=width
        CheckCuda(cudaMemcpy2DAsync(dst_y, width_, nv12->data[0],
                                    nv12->linesize[0], width_, height_,
                                    cudaMemcpyHostToDevice, cuda_stream_),
                  "cudaMemcpy2DAsync Y plane failed");

        // NV12(UV): copy width bytes per row into dst pitch=width*2 (pad half
        // row by design)
        CheckCuda(cudaMemcpy2DAsync(dst_uv, width_ * 2, nv12->data[1],
                                    nv12->linesize[1], width_, height_ / 2,
                                    cudaMemcpyHostToDevice, cuda_stream_),
                  "cudaMemcpy2DAsync UV plane failed");

        // 性能优化：等待异步操作完成（在发布前必须同步，但使用 stream
        // 可以与其他操作重叠）
        CheckCuda(cudaStreamSynchronize(cuda_stream_),
                  "cudaStreamSynchronize after copies failed");

        // 性能优化：为 NitrosImage 分配新的 GPU buffer（因为所有权会被转移）
        // 注意：NitrosImage 会接管 buffer 所有权，所以需要复制
        void *publish_buffer = nullptr;
        CheckCuda(cudaMalloc(&publish_buffer, total_size),
                  "cudaMalloc publish buffer failed");
        CheckCuda(cudaMemcpyAsync(publish_buffer, gpu_buffer_, total_size,
                                  cudaMemcpyDeviceToDevice, cuda_stream_),
                  "cudaMemcpyAsync publish buffer failed");
        CheckCuda(cudaStreamSynchronize(cuda_stream_),
                  "cudaStreamSynchronize publish buffer failed");

        // publish NitrosImage (GPU buffer ownership transferred)
        std_msgs::msg::Header header;
        header.stamp = now();
        header.frame_id = optical_frame_name_;

        nvidia::isaac_ros::nitros::NitrosImage nitros_image =
            nvidia::isaac_ros::nitros::NitrosImageBuilder()
                .WithHeader(header)
                .WithEncoding("nv12")
                .WithDimensions(static_cast<uint32_t>(height_),
                                static_cast<uint32_t>(width_))
                .WithGpuData(publish_buffer)
                .Build();
        image_pub_->publish(nitros_image);

        auto camera_info = camera_info_manager_.getCameraInfo();
        camera_info.header.stamp = header.stamp;
        camera_info.header.frame_id = camera_link_frame_name_;
        camera_info.width = width_;
        camera_info.height = height_;
        camera_info_pub_->publish(camera_info);

        // 性能优化：在处理前进行 pacing，避免累积延迟
        next_deadline += frame_period;
        std::this_thread::sleep_until(next_deadline);

        av_frame_unref(frame);
      }
    }

    RCLCPP_INFO(get_logger(), "MP4 NV12 publisher stopped.");
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_logger(), "MP4 NV12 publisher error: %s", e.what());
  }
}

} // namespace v4l2_camera
} // namespace isaac_ros
} // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
    nvidia::isaac_ros::v4l2_camera::Mp4Nv12NitrosPublisherNode)

