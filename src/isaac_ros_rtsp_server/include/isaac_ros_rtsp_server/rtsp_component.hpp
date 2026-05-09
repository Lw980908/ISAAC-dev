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

#ifndef ISAAC_ROS_RTSP_SERVER__RTSP_COMPONENT_HPP_
#define ISAAC_ROS_RTSP_SERVER__RTSP_COMPONENT_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <liveMedia.hh>
#include <mutex>
#include <string>

#define MAX_FRAME_SIZE (4 * 1024 * 1024)
// 环形缓冲区槽位数量（必须是2的幂），增加到128以应对多路并发
#define RING_BUFFER_SLOTS 128

struct FrameSlot {
  unsigned char data[MAX_FRAME_SIZE];
  int size{0};
  bool is_idr{false};
  uint64_t sequence{0};
  uint64_t stamp_ns{0};
};

class H264Or5FramedSource : public FramedSource {
public:
  static H264Or5FramedSource *createNew(UsageEnvironment &env,
                                        const char *stream_name);
  static void OnH264Or5Frame(const unsigned char *buff, int len,
                             const char *stream_name, int video_type);
  static void OnH264Or5Frame(const unsigned char *buff, int len,
                             const char *stream_name, int video_type,
                             uint64_t stamp_ns);

  static void SetTxCallback(
      const std::string &stream_name,
      const std::function<void(uint64_t stamp_ns, uint64_t tx_ns)> &callback);
  static void ClearTxCallback(const std::string &stream_name);

  static void SetStreamMaxQueueFrames(const std::string &stream_name,
                                      uint32_t max_queue_frames);

  static void SendFrame(void *client);

  static H264Or5FramedSource *frame_sources_[100];
  static std::mutex sources_mutex_;

  int SetStreamName(char *stream_name);
  char *GetStreamName();
  void SetVideoType(int video_type);

public:
  EventTriggerId eventTriggerId;
  static unsigned int referenceCount;

protected:
  explicit H264Or5FramedSource(UsageEnvironment &env);
  ~H264Or5FramedSource();

private:
  void doGetNextFrame() override;
  void doStopGettingFrames() override;

  void GetH264Or5Frame(const unsigned char *buff, int len);
  void GetH264Or5Frame(const unsigned char *buff, int len, uint64_t stamp_ns);
  bool IsIdrOrParamSet(const unsigned char *buff, int len) const;

  int AddSource(H264Or5FramedSource *source);
  int EraseSource(H264Or5FramedSource *source);

  // 环形缓冲区
  FrameSlot ring_buffer_[RING_BUFFER_SLOTS];
  std::atomic<uint64_t> write_pos_{0}; // 写位置（生产者）
  std::atomic<uint64_t> read_pos_{0};  // 读位置（消费者）
  std::atomic<uint64_t> frame_seq_{0}; // 帧序号

  // 同步
  std::mutex buffer_mu_;

  // 状态
  int video_type_{-1};
  std::atomic<bool> waiting_for_idr_{true}; // 启动时等待 IDR
  uint64_t drop_count_{0};
  uint64_t last_sent_seq_{0};
  std::atomic<uint64_t> last_tx_cb_stamp_ns_{0};

  char stream_name_[128];

  uint32_t max_queue_frames_{RING_BUFFER_SLOTS};
};

class H264ServerMediaSubsession : public OnDemandServerMediaSubsession {
public:
  static H264ServerMediaSubsession *createNew(UsageEnvironment &env,
                                              bool reuseFirstSource);

protected:
  H264ServerMediaSubsession(UsageEnvironment &env, bool reuseFirstSource);
  ~H264ServerMediaSubsession();

  FramedSource *createNewStreamSource(unsigned clientSessionId,
                                      unsigned &estBitrate) override;
  RTPSink *createNewRTPSink(Groupsock *rtpGroupsock,
                            unsigned char rtpPayloadTypeIfDynamic,
                            FramedSource *inputSource) override;
};

class H265ServerMediaSubsession : public OnDemandServerMediaSubsession {
public:
  static H265ServerMediaSubsession *createNew(UsageEnvironment &env,
                                              bool reuseFirstSource);

protected:
  H265ServerMediaSubsession(UsageEnvironment &env, bool reuseFirstSource);
  ~H265ServerMediaSubsession();

  FramedSource *createNewStreamSource(unsigned clientSessionId,
                                      unsigned &estBitrate) override;
  RTPSink *createNewRTPSink(Groupsock *rtpGroupsock,
                            unsigned char rtpPayloadTypeIfDynamic,
                            FramedSource *inputSource) override;
};

#endif // ISAAC_ROS_RTSP_SERVER__RTSP_COMPONENT_HPP_

