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

#include "isaac_ros_rtsp_server/rtsp_component.hpp"

#include <BasicUsageEnvironment.hh>
#include <InputFile.hh>
#include <liveMedia.hh>

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>

// 静态成员初始化
H264Or5FramedSource *H264Or5FramedSource::frame_sources_[100] = {nullptr};
std::mutex H264Or5FramedSource::sources_mutex_;
unsigned int H264Or5FramedSource::referenceCount = 0;

namespace {
std::mutex g_stream_cfg_mu;
std::unordered_map<std::string, uint32_t> g_stream_max_queue_frames;

std::mutex g_tx_cb_mu;
std::unordered_map<std::string,
                   std::function<void(uint64_t stamp_ns, uint64_t tx_ns)>>
    g_tx_cbs;

uint32_t ClampMaxQueueFrames(uint32_t value) {
  if (value < 1) {
    return 1;
  }
  if (value > RING_BUFFER_SLOTS) {
    return RING_BUFFER_SLOTS;
  }
  return value;
}

uint32_t GetStreamMaxQueueFrames(const std::string &stream_name) {
  std::lock_guard<std::mutex> lk(g_stream_cfg_mu);
  auto it = g_stream_max_queue_frames.find(stream_name);
  if (it == g_stream_max_queue_frames.end()) {
    return RING_BUFFER_SLOTS;
  }
  return ClampMaxQueueFrames(it->second);
}

std::function<void(uint64_t, uint64_t)> GetTxCallback(const std::string &name) {
  std::lock_guard<std::mutex> lk(g_tx_cb_mu);
  auto it = g_tx_cbs.find(name);
  if (it == g_tx_cbs.end()) {
    return {};
  }
  return it->second;
}
} // namespace

H264Or5FramedSource *H264Or5FramedSource::createNew(UsageEnvironment &env,
                                                    const char *streamName) {
  H264Or5FramedSource *source = new H264Or5FramedSource(env);
  std::snprintf(source->stream_name_, sizeof(source->stream_name_), "%s",
                streamName ? streamName : "");
  source->max_queue_frames_ =
      GetStreamMaxQueueFrames(streamName ? std::string(streamName) : "");
  return source;
}

void H264Or5FramedSource::SetStreamMaxQueueFrames(
    const std::string &stream_name, uint32_t max_queue_frames) {
  std::lock_guard<std::mutex> lk(g_stream_cfg_mu);
  g_stream_max_queue_frames[stream_name] =
      ClampMaxQueueFrames(max_queue_frames);
}

void H264Or5FramedSource::SetTxCallback(
    const std::string &stream_name,
    const std::function<void(uint64_t stamp_ns, uint64_t tx_ns)> &callback) {
  std::lock_guard<std::mutex> lk(g_tx_cb_mu);
  if (!callback) {
    g_tx_cbs.erase(stream_name);
    return;
  }
  g_tx_cbs[stream_name] = callback;
}

void H264Or5FramedSource::ClearTxCallback(const std::string &stream_name) {
  std::lock_guard<std::mutex> lk(g_tx_cb_mu);
  g_tx_cbs.erase(stream_name);
}

void H264Or5FramedSource::OnH264Or5Frame(const unsigned char *buff, int len,
                                         const char *streamName,
                                         int video_type) {
  OnH264Or5Frame(buff, len, streamName, video_type, 0);
}

void H264Or5FramedSource::OnH264Or5Frame(const unsigned char *buff, int len,
                                         const char *streamName, int video_type,
                                         uint64_t stamp_ns) {
  // 关键优化：先快速查找源，释放全局锁后再写数据
  H264Or5FramedSource *target = nullptr;
  {
    std::lock_guard<std::mutex> lk(sources_mutex_);
    constexpr int count =
        sizeof(frame_sources_) / sizeof(H264Or5FramedSource *);
    for (int i = 0; i < count; i++) {
      H264Or5FramedSource *source = frame_sources_[i];
      if (source != nullptr &&
          strcmp(source->GetStreamName(), streamName) == 0) {
        target = source;
        break;
      }
    }
  }
  // 全局锁已释放，各路可以并行写入自己的缓冲区
  if (target != nullptr) {
    target->SetVideoType(video_type);
    target->GetH264Or5Frame(buff, len, stamp_ns);
  }
}

H264Or5FramedSource::H264Or5FramedSource(UsageEnvironment &env)
    : FramedSource(env) {
  std::memset(stream_name_, 0, sizeof(stream_name_));
  std::memset(ring_buffer_, 0, sizeof(ring_buffer_));

  write_pos_.store(0, std::memory_order_relaxed);
  read_pos_.store(0, std::memory_order_relaxed);
  frame_seq_.store(0, std::memory_order_relaxed);
  waiting_for_idr_.store(true, std::memory_order_relaxed);

  drop_count_ = 0;
  last_sent_seq_ = 0;
  video_type_ = -1;

  referenceCount++;
  eventTriggerId = envir().taskScheduler().createEventTrigger(SendFrame);

  AddSource(this);
}

H264Or5FramedSource::~H264Or5FramedSource() {
  EraseSource(this);
  referenceCount--;

  if (eventTriggerId > 0) {
    envir().taskScheduler().deleteEventTrigger(eventTriggerId);
    eventTriggerId = 0;
  }
}

int H264Or5FramedSource::AddSource(H264Or5FramedSource *source) {
  std::lock_guard<std::mutex> lk(sources_mutex_);
  constexpr int count = sizeof(frame_sources_) / sizeof(H264Or5FramedSource *);
  for (int i = 0; i < count; i++) {
    if (frame_sources_[i] == nullptr) {
      frame_sources_[i] = source;
      return i;
    }
  }
  return -1;
}

int H264Or5FramedSource::EraseSource(H264Or5FramedSource *source) {
  std::lock_guard<std::mutex> lk(sources_mutex_);
  constexpr int count = sizeof(frame_sources_) / sizeof(H264Or5FramedSource *);
  for (int i = 0; i < count; i++) {
    if (frame_sources_[i] == source) {
      frame_sources_[i] = nullptr;
      return i;
    }
  }
  return -1;
}

int H264Or5FramedSource::SetStreamName(char *stream_name) {
  std::snprintf(stream_name_, sizeof(stream_name_), "%s",
                stream_name ? stream_name : "");
  return 0;
}

char *H264Or5FramedSource::GetStreamName() { return stream_name_; }

void H264Or5FramedSource::SetVideoType(int video_type) {
  if (video_type >= 0 && video_type_ < 0) {
    video_type_ = video_type;
  }
}

bool H264Or5FramedSource::IsIdrOrParamSet(const unsigned char *buff,
                                          int len) const {
  if (!buff || len <= 0) {
    return false;
  }
  int offset = 0;
  if (len >= 4 && buff[0] == 0 && buff[1] == 0 && buff[2] == 0 &&
      buff[3] == 1) {
    offset = 4;
  } else if (len >= 3 && buff[0] == 0 && buff[1] == 0 && buff[2] == 1) {
    offset = 3;
  }
  if (offset > 0) {
    buff += offset;
    len -= offset;
  }
  if (len <= 0) {
    return false;
  }
  // 未知类型时，认为是 IDR（保守策略）
  if (video_type_ < 0) {
    return true;
  }
  if (video_type_ == 1) { // H265
    const uint8_t nal_type = (buff[0] >> 1) & 0x3F;
    // IDR_W_RADL(19), IDR_N_LP(20), CRA(21), VPS(32), SPS(33), PPS(34)
    return nal_type == 19 || nal_type == 20 || nal_type == 21 ||
           nal_type == 32 || nal_type == 33 || nal_type == 34;
  }
  // H264
  const uint8_t nal_type = buff[0] & 0x1F;
  // IDR(5), SPS(7), PPS(8)
  return nal_type == 5 || nal_type == 7 || nal_type == 8;
}

void H264Or5FramedSource::GetH264Or5Frame(const unsigned char *buff, int len) {
  GetH264Or5Frame(buff, len, 0);
}

void H264Or5FramedSource::GetH264Or5Frame(const unsigned char *buff, int len,
                                          uint64_t stamp_ns) {
  if (!buff || len <= 0 || len > MAX_FRAME_SIZE) {
    drop_count_++;
    waiting_for_idr_.store(true, std::memory_order_release);
    return;
  }

  bool is_idr = IsIdrOrParamSet(buff, len);

  // 如果正在等待 IDR，只接受 IDR 帧
  if (waiting_for_idr_.load(std::memory_order_acquire)) {
    if (!is_idr) {
      return; // 静默丢弃非 IDR 帧
    }
    waiting_for_idr_.store(false, std::memory_order_release);
  }

  // 使用 try_lock 避免阻塞其他流
  std::unique_lock<std::mutex> lk(buffer_mu_, std::try_to_lock);
  if (!lk.owns_lock()) {
    // 锁被占用，丢弃这帧而不是阻塞
    drop_count_++;
    return;
  }

  uint64_t wp = write_pos_.load(std::memory_order_relaxed);
  uint64_t rp = read_pos_.load(std::memory_order_acquire);

  // 检查环形缓冲区是否已满
  if (wp - rp >= RING_BUFFER_SLOTS) {
    drop_count_++;
    // 策略：清空缓冲区，等待下一个 IDR
    read_pos_.store(wp, std::memory_order_release);
    waiting_for_idr_.store(true, std::memory_order_release);
    return;
  }

  // 写入环形缓冲区
  uint64_t slot_idx = wp & (RING_BUFFER_SLOTS - 1);
  FrameSlot &slot = ring_buffer_[slot_idx];
  memcpy(slot.data, buff, len);
  slot.size = len;
  slot.is_idr = is_idr;
  slot.sequence = frame_seq_.fetch_add(1, std::memory_order_relaxed);
  slot.stamp_ns = stamp_ns;

  // 更新写位置
  write_pos_.store(wp + 1, std::memory_order_release);

  // 释放锁后再触发事件
  lk.unlock();

  // 触发发送
  if (eventTriggerId > 0) {
    envir().taskScheduler().triggerEvent(eventTriggerId, this);
  }
}

void H264Or5FramedSource::SendFrame(void *client) {
  auto *source = static_cast<H264Or5FramedSource *>(client);
  source->doGetNextFrame();
}

void H264Or5FramedSource::doGetNextFrame() {
  if (isCurrentlyAwaitingData() == False) {
    return;
  }

  // 快速检查是否有数据（无锁）
  uint64_t rp = read_pos_.load(std::memory_order_acquire);
  uint64_t wp = write_pos_.load(std::memory_order_acquire);
  if (rp >= wp) {
    return; // 没有数据
  }

  std::unique_lock<std::mutex> lk(buffer_mu_, std::try_to_lock);
  if (!lk.owns_lock()) {
    // 锁被写入线程占用，稍后重试
    if (eventTriggerId > 0) {
      envir().taskScheduler().triggerEvent(eventTriggerId, this);
    }
    return;
  }

  // 重新读取位置（可能在等待锁时已改变）
  rp = read_pos_.load(std::memory_order_relaxed);
  wp = write_pos_.load(std::memory_order_acquire);

  if (rp >= wp) {
    return;
  }

  const uint64_t backlog = wp - rp;
  if (backlog > static_cast<uint64_t>(max_queue_frames_)) {
    rp = wp - static_cast<uint64_t>(max_queue_frames_);
    read_pos_.store(rp, std::memory_order_release);
  }

  uint64_t slot_idx = rp & (RING_BUFFER_SLOTS - 1);
  FrameSlot &slot = ring_buffer_[slot_idx];

  // 检查帧序号连续性
  if (last_sent_seq_ > 0 && slot.sequence != last_sent_seq_ + 1) {
    if (!slot.is_idr) {
      // 快进到下一个 IDR
      bool found_idr = false;
      while (rp < wp) {
        uint64_t idx = rp & (RING_BUFFER_SLOTS - 1);
        if (ring_buffer_[idx].is_idr) {
          found_idr = true;
          break;
        }
        rp++;
      }
      if (!found_idr) {
        read_pos_.store(wp, std::memory_order_release);
        waiting_for_idr_.store(true, std::memory_order_release);
        return;
      }
      read_pos_.store(rp, std::memory_order_release);
      slot_idx = rp & (RING_BUFFER_SLOTS - 1);
    }
  }

  FrameSlot &current_slot = ring_buffer_[slot_idx];

  // 检查帧大小
  if (current_slot.size > static_cast<int>(fMaxSize)) {
    drop_count_++;
    read_pos_.store(rp + 1, std::memory_order_release);
    waiting_for_idr_.store(true, std::memory_order_release);
    return;
  }

  // 复制数据
  memcpy(fTo, current_slot.data, current_slot.size);
  fFrameSize = current_slot.size;
  last_sent_seq_ = current_slot.sequence;
  const uint64_t stamp_ns = current_slot.stamp_ns;

  // 更新读位置
  read_pos_.store(rp + 1, std::memory_order_release);

  // 释放锁
  lk.unlock();

  // 设置时间戳
  if (stamp_ns != 0) {
    fPresentationTime.tv_sec = static_cast<long>(stamp_ns / 1000000000ULL);
    fPresentationTime.tv_usec =
        static_cast<long>((stamp_ns % 1000000000ULL) / 1000ULL);

    const uint64_t prev_stamp =
        last_tx_cb_stamp_ns_.load(std::memory_order_relaxed);
    if (prev_stamp != stamp_ns) {
      last_tx_cb_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      struct timeval now_tv;
      gettimeofday(&now_tv, NULL);
      const uint64_t tx_ns =
          static_cast<uint64_t>(now_tv.tv_sec) * 1000000000ULL +
          static_cast<uint64_t>(now_tv.tv_usec) * 1000ULL;
      auto cb = GetTxCallback(std::string(stream_name_));
      if (cb) {
        cb(stamp_ns, tx_ns);
      }
    }
  } else {
    gettimeofday(&fPresentationTime, NULL);
  }

  // 通知 liveMedia
  afterGetting(this);
}

void H264Or5FramedSource::doStopGettingFrames() {
  EraseSource(this);

  std::lock_guard<std::mutex> lk(buffer_mu_);
  if (eventTriggerId > 0) {
    envir().taskScheduler().deleteEventTrigger(eventTriggerId);
    eventTriggerId = 0;
  }

  write_pos_.store(0, std::memory_order_relaxed);
  read_pos_.store(0, std::memory_order_relaxed);
  waiting_for_idr_.store(true, std::memory_order_relaxed);

  FramedSource::doStopGettingFrames();
}

H264ServerMediaSubsession *
H264ServerMediaSubsession::createNew(UsageEnvironment &env,
                                     bool reuseFirstSource) {
  auto *subsession = new H264ServerMediaSubsession(env, reuseFirstSource);
  return subsession;
}

H264ServerMediaSubsession::H264ServerMediaSubsession(UsageEnvironment &env,
                                                     bool reuseFirstSource)
    : OnDemandServerMediaSubsession(env, reuseFirstSource) {}

H264ServerMediaSubsession::~H264ServerMediaSubsession() {}

FramedSource *
H264ServerMediaSubsession::createNewStreamSource(unsigned clientSessionId,
                                                 unsigned &estBitrate) {
  (void)clientSessionId;
  estBitrate = 8000;
  FramedSource *source =
      H264Or5FramedSource::createNew(envir(), fParentSession->streamName());
  source = H264VideoStreamDiscreteFramer::createNew(envir(), source);
  return source;
}

RTPSink *H264ServerMediaSubsession::createNewRTPSink(
    Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic,
    FramedSource *inputSource) {
  (void)inputSource;
  return H264VideoRTPSink::createNew(envir(), rtpGroupsock,
                                     rtpPayloadTypeIfDynamic);
}

H265ServerMediaSubsession *
H265ServerMediaSubsession::createNew(UsageEnvironment &env,
                                     bool reuseFirstSource) {
  auto *subsession = new H265ServerMediaSubsession(env, reuseFirstSource);
  return subsession;
}

H265ServerMediaSubsession::H265ServerMediaSubsession(UsageEnvironment &env,
                                                     bool reuseFirstSource)
    : OnDemandServerMediaSubsession(env, reuseFirstSource) {}

H265ServerMediaSubsession::~H265ServerMediaSubsession() {}

FramedSource *
H265ServerMediaSubsession::createNewStreamSource(unsigned clientSessionId,
                                                 unsigned &estBitrate) {
  (void)clientSessionId;
  estBitrate = 8000;
  FramedSource *source =
      H264Or5FramedSource::createNew(envir(), fParentSession->streamName());
  source = H265VideoStreamDiscreteFramer::createNew(envir(), source);
  return source;
}

RTPSink *H265ServerMediaSubsession::createNewRTPSink(
    Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic,
    FramedSource *inputSource) {
  (void)inputSource;
  return H265VideoRTPSink::createNew(envir(), rtpGroupsock,
                                     rtpPayloadTypeIfDynamic);
}

