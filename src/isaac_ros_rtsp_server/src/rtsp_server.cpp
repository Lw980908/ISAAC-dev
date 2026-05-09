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

#include "isaac_ros_rtsp_server/rtsp_server.hpp"

#include <BasicUsageEnvironment.hh>
#include <liveMedia.hh>

#include <condition_variable>
#include <map>
#include <mutex>
#include <utility>

#include "isaac_ros_rtsp_server/rtsp_component.hpp"

namespace rtspcomponent {
namespace {
struct SharedRtspServer {
  std::mutex mu;
  std::condition_variable cv;
  bool ready = false;
  bool stopping = false;
  int ref_count = 0;

  std::shared_ptr<RtspServerConfig> sp_config = nullptr;
  RTSPServer *server = nullptr;
  TaskScheduler *scheduler = nullptr;
  UsageEnvironment *env = nullptr;
  std::shared_ptr<std::thread> thread = nullptr;
  EventLoopWatchVariable watch_variable = 0;

  std::map<std::string, VideoType> sessions;

  struct AddSessionArgs {
    SharedRtspServer *self{};
    std::string stream_name;
    VideoType video_type{VideoType::H264};
    uint32_t max_queue_frames{128};
  };

  static void AddSessionTask(void *clientData) {
    auto *args = static_cast<AddSessionArgs *>(clientData);
    SharedRtspServer *self = args ? args->self : nullptr;
    const std::string stream_name = args ? args->stream_name : std::string();
    const VideoType video_type = args ? args->video_type : VideoType::H264;
    const uint32_t max_queue_frames = args ? args->max_queue_frames : 128;
    delete args;
    if (!self || !self->env || !self->server) {
      return;
    }

    {
      std::lock_guard<std::mutex> lk(self->mu);
      if (self->sessions.find(stream_name) != self->sessions.end()) {
        return;
      }
      self->sessions.emplace(stream_name, video_type);
    }

    ServerMediaSession *sms = ServerMediaSession::createNew(
        *self->env, stream_name.c_str(), "rtsp server", "session");
    H264Or5FramedSource::SetStreamMaxQueueFrames(stream_name, max_queue_frames);
    if (video_type == VideoType::H264) {
      sms->addSubsession(
          H264ServerMediaSubsession::createNew(*self->env, True));
    } else if (video_type == VideoType::H265) {
      sms->addSubsession(
          H265ServerMediaSubsession::createNew(*self->env, True));
    }
    self->server->addServerMediaSession(sms);

    char *url = self->server->rtspURL(sms);
    *self->env << "\n Play this stream using the URL: \n\t" << url << "\n\n";
    delete[] url;
  }

  static void StopServerTask(void *clientData) {
    auto *self = static_cast<SharedRtspServer *>(clientData);
    if (!self) {
      return;
    }
    self->watch_variable = 1;
  }
};

static std::mutex g_servers_mu;
static std::map<int, std::shared_ptr<SharedRtspServer>> g_servers;

static std::shared_ptr<SharedRtspServer>
getOrCreateShared(int port, std::shared_ptr<RtspServerConfig> base_cfg) {
  std::lock_guard<std::mutex> lk(g_servers_mu);
  auto it = g_servers.find(port);
  if (it != g_servers.end()) {
    return it->second;
  }

  auto ss = std::make_shared<SharedRtspServer>();
  ss->sp_config = std::make_shared<RtspServerConfig>();
  if (base_cfg) {
    *ss->sp_config = *base_cfg;
  }
  ss->sp_config->port_ = port;
  g_servers.emplace(port, ss);
  return ss;
}
} // namespace

RtspServer::RtspServer(std::shared_ptr<RtspServerConfig> sp_config) {
  sp_config_ = std::make_shared<RtspServerConfig>();
  if (sp_config) {
    *sp_config_ = *sp_config;
  }
}

RtspServer::~RtspServer() {}

int RtspServer::Init() {
  rtsp_server_thread_ = nullptr;
  watch_variable_ = 0;
  return 0;
}

int RtspServer::Start() {
  auto shared = getOrCreateShared(sp_config_->port_, sp_config_);
  {
    std::lock_guard<std::mutex> lk(shared->mu);
    shared->ref_count++;
    if (!shared->thread) {
      shared->watch_variable = 0;
      shared->stopping = false;
      shared->thread = std::make_shared<std::thread>([shared]() {
        shared->scheduler = BasicTaskScheduler::createNew();
        shared->env = BasicUsageEnvironment::createNew(*shared->scheduler);

        UserAuthenticationDatabase *authDB = NULL;
        if (shared->sp_config && shared->sp_config->auth_mode_) {
          authDB = new UserAuthenticationDatabase;
          authDB->addUserRecord(shared->sp_config->user_.c_str(),
                                shared->sp_config->password_.c_str());
        }

        // 增大 RTP 发送缓冲区以支持多路高分辨率
        OutPacketBuffer::maxSize = 2 * 1024 * 1024;

        portNumBits rtspServerPortNum =
            shared->sp_config ? shared->sp_config->port_ : 0;
        shared->server =
            RTSPServer::createNew(*shared->env, rtspServerPortNum, authDB);
        if (shared->server == NULL) {
          std::lock_guard<std::mutex> lk2(shared->mu);
          shared->ready = false;
          shared->stopping = true;
          shared->cv.notify_all();
          return;
        }

        {
          std::lock_guard<std::mutex> lk2(shared->mu);
          shared->ready = true;
          shared->cv.notify_all();
        }

        shared->env->taskScheduler().doEventLoop(&shared->watch_variable);

        Medium::close(shared->server);
        shared->env->reclaim();
        delete shared->scheduler;
        shared->server = NULL;
        shared->env = NULL;
        shared->scheduler = NULL;
      });
    }
  }

  {
    std::unique_lock<std::mutex> lk(shared->mu);
    shared->cv.wait(lk, [&]() { return shared->ready || shared->stopping; });
    if (!shared->ready) {
      return -1;
    }
  }

  {
    std::lock_guard<std::mutex> lk(shared->mu);
    if (shared->env) {
      auto *args = new SharedRtspServer::AddSessionArgs{
          shared.get(), sp_config_->stream_name, sp_config_->video_type_,
          sp_config_->max_queue_frames};
      shared->env->taskScheduler().scheduleDelayedTask(
          0, SharedRtspServer::AddSessionTask, args);
    }
  }
  return 0;
}

int RtspServer::Stop() {
  auto shared = getOrCreateShared(sp_config_->port_, sp_config_);
  bool need_stop = false;
  {
    std::lock_guard<std::mutex> lk(shared->mu);
    if (shared->ref_count > 0) {
      shared->ref_count--;
    }
    if (shared->ref_count == 0 && shared->env && shared->thread) {
      need_stop = true;
      shared->stopping = true;
      shared->ready = false;
      shared->env->taskScheduler().scheduleDelayedTask(
          0, SharedRtspServer::StopServerTask, shared.get());
    }
  }
  if (need_stop) {
    if (shared->thread && shared->thread->joinable()) {
      shared->thread->join();
    }
    std::lock_guard<std::mutex> lk(shared->mu);
    shared->thread = nullptr;
    shared->sessions.clear();
    shared->stopping = false;
  }
  return 0;
}

int RtspServer::DeInit() { return 0; }

int RtspServer::SendData(const unsigned char *buf, int buf_len, int media_type,
                         uint64_t stamp_ns) {
  if (media_type == static_cast<int>(VideoType::H264) ||
      media_type == static_cast<int>(VideoType::H265)) {
    if (buf_len >= 4 && buf[0] == 0 && buf[1] == 0 && buf[2] == 0 &&
        buf[3] == 1) {
      H264Or5FramedSource::OnH264Or5Frame(
          buf + 4, buf_len - 4, sp_config_->stream_name.c_str(), media_type,
          stamp_ns);
    } else if (buf_len >= 3 && buf[0] == 0 && buf[1] == 0 && buf[2] == 1) {
      H264Or5FramedSource::OnH264Or5Frame(
          buf + 3, buf_len - 3, sp_config_->stream_name.c_str(), media_type,
          stamp_ns);
    }
  }
  return 0;
}

int RtspServer::RTSPServerRun() { return 0; }

} // namespace rtspcomponent
