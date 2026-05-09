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

#ifndef ISAAC_ROS_RTSP_SERVER__RTSP_SERVER_HPP_
#define ISAAC_ROS_RTSP_SERVER__RTSP_SERVER_HPP_

#include <cstdint>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

class RTSPServer;
class TaskScheduler;
class UsageEnvironment;
class ServerMediaSession;

namespace rtspcomponent {

enum class VideoType { H264 = 0, H265 };

struct RtspServerConfig {
  std::string stream_name{"chn0"};
  int port_ = 8554;
  VideoType video_type_{VideoType::H264};

  uint8_t auth_mode_ = 0;
  std::string user_;
  std::string password_;

  uint32_t max_queue_frames = 128;
};

class RtspServer {
public:
  explicit RtspServer(std::shared_ptr<RtspServerConfig> sp_config);
  ~RtspServer();
  int Init();
  int Start();
  int Stop();
  int DeInit();
  int SendData(const unsigned char *buf, int buf_len, int media_type,
               uint64_t stamp_ns = 0);

private:
  int RTSPServerRun();

  std::shared_ptr<RtspServerConfig> sp_config_ = nullptr;
  RTSPServer *rtsp_server_{};
  TaskScheduler *scheduler_{};
  UsageEnvironment *env_{};
  ServerMediaSession *server_media_session_{};
  std::shared_ptr<std::thread> rtsp_server_thread_{};
  volatile char watch_variable_{};
};

} // namespace rtspcomponent

#endif // ISAAC_ROS_RTSP_SERVER__RTSP_SERVER_HPP_
