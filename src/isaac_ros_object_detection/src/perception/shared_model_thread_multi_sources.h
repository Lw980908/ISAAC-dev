#include "../distortion_correction/undistorter_interface.h"
#include "thread_interface.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>
#include <vector>

// NITROS 类型支持
#include "gxf_nv12_draw.h"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include <cuda_runtime.h>

// NV12 发布 buffer 池：避免每帧 cudaMalloc/cudaFree 抖动
struct Nv12PublishPool {
  explicit Nv12PublishPool(size_t bytes) : size_bytes(bytes) {}
  Nv12PublishPool(const Nv12PublishPool &) = delete;
  Nv12PublishPool &operator=(const Nv12PublishPool &) = delete;

  ~Nv12PublishPool() {
    std::lock_guard<std::mutex> lk(m);
    for (void *p : free_list) {
      if (p) {
        cudaFree(p);
      }
    }
    free_list.clear();
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

  // 预热：提前分配若干 buffer，避免运行时 cudaMalloc 引起的全局同步抖动
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

  void release(void *p) {
    if (!p) {
      return;
    }
    std::lock_guard<std::mutex> lk(m);
    free_list.push_back(p);
  }

  size_t size_bytes = 0;
  std::mutex m;
  std::vector<void *> free_list;
};

class SharedModelMultiSourcesPipelineManager : public PipelineInterface {
private:
  // 常量配置
  static constexpr int MAX_QUEUE_SIZE = 2;
  // NITROS NV12 预分配池槽位数（每路）：应明显大于 2，避免 publish/detect/回调
  // 三方并发导致“池满丢帧/抖动”。槽位越大，丢帧越少，但占用更多显存。
  static constexpr int NITROS_NV12_POOL_SLOTS = 8;

  // 每个输入源的上下文
  struct InputContext {
    std::shared_ptr<Nv12PublishPool> nitros_publish_pool_;

    std::unique_ptr<UndistorterInterface> undistorter_;

    // sensor_msgs::msg::Image 发布器
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>>
        image_raw_publisher_;

    // NITROS 发布器（用于发布 NV12 图像给 H264 编码器）
    std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>>
        nitros_image_publisher_;

    // CUDA BGR 缓冲区（解码/发布复用，避免频繁 malloc/free）
    void *cuda_bgr_decode_buffer_ = nullptr;
    size_t cuda_bgr_decode_buffer_size_ = 0;
    void *cuda_bgr_publish_buffer_ = nullptr;
    size_t cuda_bgr_publish_buffer_size_ = 0;
    // GPU bbox 缓冲（发布线程复用，避免每帧 cudaMalloc/cudaFree）
    void *cuda_bbox_buffer_ = nullptr;
    size_t cuda_bbox_buffer_size_ = 0;

    // CUDA 流（每个源独立，避免串行化）
    cudaStream_t cuda_stream_ = nullptr;
    // CUDA 流：用于发布/叠框（避免与回调 copy 使用同一 stream 而互相阻塞）
    cudaStream_t cuda_publish_stream_ = nullptr;

    std::queue<std::unique_ptr<cv::Mat>> raw_frame_queue;
    std::queue<std::unique_ptr<cv::Mat>> detection_queue;
    std::queue<cv::Mat> display_queue;
    std::queue<std::vector<uint8_t>> decode_queue_;

    // NITROS GPU NV12 数据队列：
    // - detect_queue: 给共享模型检测线程使用（轮询）
    // - publish_queue: 给每路发布线程使用（保证帧率，叠历史框）
    struct GpuNV12Data {
      void *gpu_buffer; // GPU 缓冲区（需要 cudaFree）
      int width;
      int height;
      int y_stride;
      int uv_stride;
      size_t total_size{0}; // NV12 总字节数（用于调试/校验）
      cudaEvent_t ready_event{nullptr}; // copy 完成事件（避免回调中同步阻塞）
      // 如果来自“预分配池”，则 decode 完成后归还该槽位（不做
      // cudaFree/cudaEventDestroy）
      int pool_index{-1};
      int32_t stamp_sec{0};
      uint32_t stamp_nanosec{0};
    };
    // 发布队列（保持历史命名，避免大范围改动）：每路只保留最新一帧用于叠框发布
    std::queue<GpuNV12Data> nitros_gpu_queue_;
    std::mutex nitros_gpu_queue_mutex_;
    std::condition_variable nitros_gpu_cond_;

    // 检测队列：每路只保留最新一帧用于共享模型轮询检测
    std::queue<GpuNV12Data> nitros_detect_queue_;
    std::mutex nitros_detect_queue_mutex_;

    // 预分配 NV12 GPU buffer 池：避免回调里每帧 cudaMalloc/cudaFree
    // 造成同步抖动
    struct Nv12PoolSlot {
      void *gpu_buffer{nullptr};
      size_t size_bytes{0};
      cudaEvent_t ready_event{nullptr};
      bool in_use{false};
      int ref_count{0}; // 被 detect/publish 引用的次数
    };
    std::vector<Nv12PoolSlot> nitros_nv12_pool_;
    std::mutex nitros_nv12_pool_mutex_;
    // 当回调线程因队列满而丢弃旧帧时，把对应槽位放到 GC 队列，等待事件 ready
    // 后归还
    std::queue<int> nitros_nv12_pool_gc_;
    std::mutex nitros_nv12_pool_gc_mutex_;

    // 当回调线程需要丢帧时，不要在回调里
    // cudaEventSynchronize/cudaFree（会抖帧）。 统一把旧 buffer/event 放到 GC
    // 队列，由解码线程异步回收。
    std::queue<GpuNV12Data> nitros_gpu_gc_queue_;
    std::mutex nitros_gpu_gc_mutex_;

    std::mutex raw_frame_mutex_;
    std::mutex detection_mutex_;
    std::mutex display_mutex_;
    std::mutex image_processing_mutex;
    std::mutex publish_mutex_;
    std::mutex decode_queue_mutex_;
    std::mutex undistorter_mutex_;
    std::mutex last_detection_mutex;
    std::condition_variable raw_frame_cond_;
    std::condition_variable detection_cond_;
    std::condition_variable display_cond_;
    std::condition_variable decode_cond_;

    // 轮询检测的“最新帧入队时间”，用于统计检测等待时间（从入队到开始检测）
    std::chrono::steady_clock::time_point last_detection_enqueue_tp_;
    bool has_last_detection_enqueue_tp_{false};

    // 轮询检测统计：帮助定位 wait_ms/丢帧/空取帧
    uint64_t detection_dequeued_{0}; // 检测线程成功取到帧的次数
    uint64_t ready_but_empty_{0}; // 选中source后却取不到帧（竞态/覆盖）

    // 历史检测结果
    std::vector<std::vector<Box>> last_detection_boxes;

    std::vector<GxfBBox> nv12_people_bbs_host_;
    std::vector<GxfCircleMark> nv12_marks_host_;

    // ===== FPS（用于叠加到发布图像右上角）=====
    std::chrono::steady_clock::time_point fps_window_start_tp_{
        std::chrono::steady_clock::now()};
    uint32_t fps_window_frames_{0};
    double fps_value_{0.0}; // 每秒帧率（窗口统计）

    // 配置参数
    ModelConfig model_config_;
    InputConfig input_config_;
    ToothDetectionConfig tooth_detection_config_;
    PedestrianDetectionConfig pedestrian_detection_config_;
    ImageProcessingConfig image_processing_config_;
    ROIConfig roi_config_;
    MqttConfig mqtt_config_;
    UndistortConfig undistort_config_;
    ImageEncoderDecoderConfig image_encoder_decoder_config_;

    // 自定义构造函数
    explicit InputContext(
        const ModelConfig &model_config, const InputConfig &input_config,
        const ToothDetectionConfig &tooth_detection_config,
        const PedestrianDetectionConfig &pedestrian_detection_config,
        const ImageProcessingConfig &image_processing_config,
        const ROIConfig &roi_config_, const MqttConfig &mqtt_config_,
        const UndistortConfig &undistort_config,
        const ImageEncoderDecoderConfig &image_encoder_decoder_config)
        : model_config_(model_config), input_config_(input_config),
          tooth_detection_config_(tooth_detection_config),
          pedestrian_detection_config_(pedestrian_detection_config),
          image_processing_config_(image_processing_config),
          roi_config_(roi_config_), mqtt_config_(mqtt_config_),
          undistort_config_(undistort_config),
          image_encoder_decoder_config_(image_encoder_decoder_config),
          last_detection_boxes(), cuda_bgr_decode_buffer_(nullptr),
          cuda_bgr_decode_buffer_size_(0), cuda_bgr_publish_buffer_(nullptr),
          cuda_bgr_publish_buffer_size_(0), cuda_stream_(nullptr),
          cuda_publish_stream_(nullptr) {
      // 创建独立的 CUDA 流（允许多源并行 GPU 操作）
      cudaStreamCreate(&cuda_stream_);
      cudaStreamCreateWithFlags(&cuda_publish_stream_, cudaStreamNonBlocking);
    }

    InputContext() = default;

    ~InputContext() {
      // 释放 CUDA 缓冲区
      if (cuda_bgr_decode_buffer_) {
        cudaFree(cuda_bgr_decode_buffer_);
        cuda_bgr_decode_buffer_ = nullptr;
      }
      if (cuda_bgr_publish_buffer_) {
        cudaFree(cuda_bgr_publish_buffer_);
        cuda_bgr_publish_buffer_ = nullptr;
      }
      if (cuda_bbox_buffer_) {
        cudaFree(cuda_bbox_buffer_);
        cuda_bbox_buffer_ = nullptr;
      }
      if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
      }
      if (cuda_publish_stream_) {
        cudaStreamDestroy(cuda_publish_stream_);
        cuda_publish_stream_ = nullptr;
      }
    }
  };

  size_t num_sources{0};
  std::unordered_map<int, InputContext> input_contexts_;

  // NITROS 订阅器（每个源一个）
  std::vector<
      std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
          nvidia::isaac_ros::nitros::NitrosImageView>>>
      nitros_subscribers_;

  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>
      image_raw_subscribers_;

  // 全局状态
  std::atomic<bool> is_running_{true};
  std::vector<std::thread> worker_threads_;
  std::thread detection_thread_;
  std::thread ros_executor_thread_;
  std::thread gui_thread_;
  std::vector<std::thread> decode_threads_;

  // 互斥锁
  std::mutex yolo_mutex_;
  std::mutex second_yolo_mutex_;
  std::mutex trt_classifier_mutex_;
  std::mutex mqtt_publisher_mutex_;
  std::mutex ros2_client_mutex_;
  std::mutex global_detection_mutex_;

  // 轮询检测调度：用“有帧可检测”的源集合替代忙轮询
  std::mutex ready_mutex_;
  std::condition_variable ready_cond_;
  std::vector<bool> source_ready_flags_;

  // ROS2执行器
  rclcpp::executors::MultiThreadedExecutor::SharedPtr executor_;
  std::vector<rclcpp::CallbackGroup::SharedPtr> callback_groups_;

  // 内部处理函数
  void processUndistortAndDraw(int source_id);
  void processDetectionGlobal();
  void
  processImageRawDecoding(int source_id); // sensor_msgs::msg::Image 解码处理
  void GUI();

  // NITROS 图像处理函数
  // 将 BGR cv::Mat 转换为 NV12 并发布为 NITROS 消息
  bool publishNitrosImage(InputContext &ctx, const cv::Mat &bgr_frame,
                          int source_id);

  // 辅助方法
  void initResources() override;
  void cleanupResources();
  // 入队原始帧：
  // - const& 版本会 clone（适用于相机直采/复用缓冲的场景）
  // - && 版本会 move（适用于解码线程新建帧，避免额外拷贝）
  void enqueueRawFrame(int source_id, const cv::Mat &frame);
  void enqueueRawFrame(int source_id, cv::Mat &&frame);
  cv::Mat dequeueDisplayFrame(int source_id);
  void markSourceReadyForDetection(int source_id);
  bool anySourceReadyForDetectionLocked() const;

public:
  explicit SharedModelMultiSourcesPipelineManager(
      const InputConfig &input_config);
  SharedModelMultiSourcesPipelineManager();
  ~SharedModelMultiSourcesPipelineManager() override;
  void start() override;
  void stop() override;
  bool isRunning() const override;
};
