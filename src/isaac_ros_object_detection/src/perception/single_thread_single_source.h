#include "thread_interface.h"
#include <cuda_runtime.h>
#include <rclcpp/callback_group.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <opencv2/core/mat.hpp>

class SingleThreadSingleSourcePipelineManager : public PipelineInterface {
public:
  struct Nv12PublishPool;
  struct Nv12SubPool;

  struct Nv12GpuFrame {
    void *gpu_buffer{nullptr};
    int width{0};
    int height{0};
    int y_stride{0};
    int uv_stride{0};
    size_t size_bytes{0};
    cudaEvent_t ready_event{nullptr};
    int pool_index{-1};
    int32_t stamp_sec{0};
    uint32_t stamp_nanosec{0};
  };

private:
  void initResources() override;
  void cleanupResources();

  void detectionThread();
  void drawThread();
  bool publishNitrosImage(const cv::Mat &bgr_frame);
  bool
  publishNitrosNv12WithOverlay(const Nv12GpuFrame &frame,
                               const std::vector<std::vector<Box>> &bboxes);

  int num_sources{4};                             // 输入源数量
  int process_frame_{0};                          // 处理帧计数
  std::vector<std::vector<Box>> previous_bboxes_; // 缓存上一次的检测结果

  // 线程相关
  std::thread detection_thread_;
  std::thread draw_thread_;
  std::thread ros_spin_thread_;
  bool running_ = true;

  // 图像队列
  size_t max_queue_size_{3};
  std::queue<cv::Mat> image_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  std::queue<Nv12GpuFrame> nv12_queue_;
  std::mutex nv12_queue_mutex_;
  std::condition_variable nv12_queue_cv_;
  std::shared_ptr<Nv12SubPool> nitros_sub_pool_;

  rclcpp::CallbackGroup::SharedPtr callback_group_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_raw_subscriber_;

  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>>
      image_raw_publisher_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>>
      nitros_image_publisher_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosImageView>>
      nitros_image_subscriber_;

  void *cuda_nv12_sub_buffer_ = nullptr;
  size_t cuda_nv12_sub_buffer_size_ = 0;
  void *cuda_bgr_sub_buffer_ = nullptr;
  size_t cuda_bgr_sub_buffer_size_ = 0;
  void *cuda_bgr_publish_buffer_ = nullptr;
  size_t cuda_bgr_publish_buffer_size_ = 0;
  void *cuda_bbox_buffer_ = nullptr;
  size_t cuda_bbox_buffer_size_ = 0;
  cudaStream_t cuda_stream_ = nullptr;
  cudaStream_t cuda_sub_stream_ = nullptr;
  cudaStream_t cuda_publish_stream_ = nullptr;

  std::shared_ptr<Nv12PublishPool> nitros_publish_pool_;
  std::mutex trt_classifier_mutex_;
  std::mutex mqtt_publisher_mutex_;

  struct DrawItem {
    bool is_nv12{false};
    Nv12GpuFrame nv12_frame;
    cv::Mat bgr_frame;
    std::vector<std::vector<Box>> bboxes;
    std::shared_ptr<const ToothDetectionConfig> tooth_detection_config;
    std::shared_ptr<const PedestrianDetectionConfig>
        pedestrian_detection_config;
  };

  size_t max_draw_queue_size_{2};
  std::queue<DrawItem> draw_queue_;
  std::mutex draw_queue_mutex_;
  std::condition_variable draw_queue_cv_;

  std::mutex last_detection_mutex_;
  std::vector<std::vector<Box>> last_detection_boxes_;

  std::chrono::steady_clock::time_point fps_window_start_tp_{
      std::chrono::steady_clock::now()};
  uint32_t fps_window_frames_{0};
  double fps_value_{0.0};

public:
  // 构造/析构
  explicit SingleThreadSingleSourcePipelineManager(
      const InputConfig &input_config);
  SingleThreadSingleSourcePipelineManager();
  ~SingleThreadSingleSourcePipelineManager() override;

  // 核心接口
  void start() override;
  void stop() override;
  bool isRunning() const override;
};
