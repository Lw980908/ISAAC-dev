#include "../distortion_correction/undistorter_interface.h"
#include "thread_interface.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
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

// 每个源独立的检测模型需要完整类型定义
#include "MobileNetV2/classify.h"
#include "yolo_model.h"

struct MultiThreadsNv12PublishPool {
  explicit MultiThreadsNv12PublishPool(size_t bytes) : size_bytes(bytes) {}
  MultiThreadsNv12PublishPool(const MultiThreadsNv12PublishPool &) = delete;
  MultiThreadsNv12PublishPool &
  operator=(const MultiThreadsNv12PublishPool &) = delete;

  ~MultiThreadsNv12PublishPool() {
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

class MultiThreadsMultiSourcesPipelineManager : public PipelineInterface {
private:
  // 常量配置
  static constexpr int MAX_QUEUE_SIZE = 2;
  static constexpr int NITROS_NV12_POOL_SLOTS = 8;

  // 每个输入源的上下文
  struct InputContext {
    std::unique_ptr<UndistorterInterface> undistorter_;

    // 每个源独立的检测模型（不再共享）
    std::unique_ptr<YOLOModel> yolo_;
    std::unique_ptr<YOLOModel> second_yolo_; // 二次检测模型（可选）
    std::unique_ptr<TensorRTInference> trt_classifier_; // 分类器（可选）
    std::mutex yolo_mutex_;
    std::mutex second_yolo_mutex_;
    std::mutex trt_classifier_mutex_;

    // sensor_msgs::msg::Image 发布器
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>>
        image_raw_publisher_;

    // NITROS 发布器（用于发布 NV12 图像给 H264 编码器）
    std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>>
        nitros_image_publisher_;

    // CUDA 缓冲区（用于 NITROS 发布）
    void *cuda_nv12_buffer_ = nullptr;
    size_t cuda_nv12_buffer_size_ = 0;

    // CUDA 流（每个源独立，避免串行化）
    cudaStream_t cuda_stream_ = nullptr;
    cudaStream_t cuda_publish_stream_ = nullptr;
    void *cuda_bbox_buffer_ = nullptr;
    size_t cuda_bbox_buffer_size_ = 0;
    void *cuda_bgr_publish_buffer_ = nullptr;
    size_t cuda_bgr_publish_buffer_size_ = 0;

    std::shared_ptr<MultiThreadsNv12PublishPool> nitros_publish_pool_;

    std::queue<std::unique_ptr<cv::Mat>> raw_frame_queue;
    std::queue<std::unique_ptr<cv::Mat>> detection_queue;
    std::queue<cv::Mat> display_queue;
    std::queue<std::vector<uint8_t>> decode_queue_;

    // NITROS GPU NV12 数据队列（从 NITROS 订阅器接收，在独立线程中转换）
    struct GpuNV12Data {
      void *gpu_buffer;
      int width;
      int height;
      int y_stride;
      int uv_stride;
      size_t total_size{0};
      cudaEvent_t ready_event{nullptr};
      int pool_index{-1};
      int32_t stamp_sec{0};
      uint32_t stamp_nanosec{0};
    };
    std::queue<GpuNV12Data> nitros_gpu_queue_;
    std::mutex nitros_gpu_queue_mutex_;
    std::condition_variable nitros_gpu_cond_;

    std::queue<GpuNV12Data> nitros_detect_queue_;
    std::mutex nitros_detect_queue_mutex_;
    std::condition_variable nitros_detect_cond_;

    std::queue<GpuNV12Data> nitros_publish_queue_;
    std::mutex nitros_publish_queue_mutex_;
    std::condition_variable nitros_publish_cond_;

    struct Nv12PoolSlot {
      void *gpu_buffer{nullptr};
      size_t size_bytes{0};
      cudaEvent_t ready_event{nullptr};
      bool in_use{false};
      int ref_count{0};
    };
    std::vector<Nv12PoolSlot> nitros_nv12_pool_;
    std::mutex nitros_nv12_pool_mutex_;
    std::queue<int> nitros_nv12_pool_gc_;
    std::mutex nitros_nv12_pool_gc_mutex_;

    // NITROS 图像队列（已转换为 BGR 的图像，保留兼容性）
    std::queue<cv::Mat> nitros_decode_queue_;
    std::mutex nitros_decode_queue_mutex_;
    std::condition_variable nitros_decode_cond_;

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

    // 历史检测结果
    std::vector<std::vector<Box>> last_detection_boxes;
    std::vector<GxfBBox> nv12_people_bbs_host_;
    std::vector<GxfCircleMark> nv12_marks_host_;

    std::chrono::steady_clock::time_point fps_window_start_tp_{
        std::chrono::steady_clock::now()};
    uint32_t fps_window_frames_{0};
    double fps_value_{0.0};

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
          last_detection_boxes(), cuda_nv12_buffer_(nullptr),
          cuda_nv12_buffer_size_(0), cuda_stream_(nullptr), yolo_(nullptr),
          second_yolo_(nullptr), trt_classifier_(nullptr) {
      // 创建独立的 CUDA 流（允许多源并行 GPU 操作）
      cudaStreamCreate(&cuda_stream_);
      cudaStreamCreateWithFlags(&cuda_publish_stream_, cudaStreamNonBlocking);
    }

    InputContext() = default;

    ~InputContext() {
      // 释放 CUDA 缓冲区
      if (cuda_nv12_buffer_) {
        cudaFree(cuda_nv12_buffer_);
        cuda_nv12_buffer_ = nullptr;
      }
      if (cuda_bbox_buffer_) {
        cudaFree(cuda_bbox_buffer_);
        cuda_bbox_buffer_ = nullptr;
      }
      if (cuda_bgr_publish_buffer_) {
        cudaFree(cuda_bgr_publish_buffer_);
        cuda_bgr_publish_buffer_ = nullptr;
      }
      if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
      }
      if (cuda_publish_stream_) {
        cudaStreamDestroy(cuda_publish_stream_);
        cuda_publish_stream_ = nullptr;
      }
      for (auto &slot : nitros_nv12_pool_) {
        if (slot.ready_event) {
          cudaEventDestroy(slot.ready_event);
          slot.ready_event = nullptr;
        }
        if (slot.gpu_buffer) {
          cudaFree(slot.gpu_buffer);
          slot.gpu_buffer = nullptr;
        }
      }
      nitros_nv12_pool_.clear();
      // 注意：yolo_, second_yolo_, trt_classifier_ 由 unique_ptr 自动释放
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
  std::vector<std::thread> detection_threads_; // 每个源独立的检测线程
  std::vector<std::thread> nv12_publish_threads_;
  std::thread ros_executor_thread_;
  std::thread gui_thread_;
  std::vector<std::thread> decode_threads_;

  // 互斥锁（仅保留全局资源的锁）
  std::mutex mqtt_publisher_mutex_;
  std::mutex ros2_client_mutex_;

  // ROS2执行器
  rclcpp::executors::MultiThreadedExecutor::SharedPtr executor_;
  std::vector<rclcpp::CallbackGroup::SharedPtr> callback_groups_;

  // 内部处理函数
  void processUndistortAndDraw(int source_id);
  void processDetection(int source_id); // 每个源独立的检测线程
  void
  processImageRawDecoding(int source_id); // sensor_msgs::msg::Image 解码处理
  void processNitrosDecoding(int source_id); // NITROS NV12 解码处理
  void processNitrosOverlayPublish(int source_id);
  void GUI();

  // NITROS 图像处理函数
  // 将 NV12 GPU 数据转换为 BGR cv::Mat
  cv::Mat
  convertNV12ToBGR(const nvidia::isaac_ros::nitros::NitrosImageView &view);
  // 将 BGR cv::Mat 转换为 NV12 并发布为 NITROS 消息
  bool publishNitrosImage(InputContext &ctx, const cv::Mat &bgr_frame,
                          int source_id);
  // 分配 CUDA NV12 缓冲区
  bool allocateCudaNV12Buffer(InputContext &ctx, int width, int height);

  // 辅助方法
  void initResources() override;
  void cleanupResources();
  void enqueueRawFrame(int source_id, const cv::Mat &frame);
  void enqueueRawFrame(int source_id, cv::Mat &&frame);
  cv::Mat dequeueDisplayFrame(int source_id);

public:
  explicit MultiThreadsMultiSourcesPipelineManager(
      const InputConfig &input_config);
  MultiThreadsMultiSourcesPipelineManager();
  ~MultiThreadsMultiSourcesPipelineManager() override;
  void start() override;
  void stop() override;
  bool isRunning() const override;
};
