#include "thread_interface.h"
#include <cuda_runtime.h>
#include <rclcpp/callback_group.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "distance_types.h"

class CameraCombinedWithLidarPipelineManager : public PipelineInterface
{
public:
  struct Nv12PublishPool;
  struct Nv12SubPool;

  struct Nv12GpuFrame
  {
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

  void handlePointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void
  handleLivoxCustomMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg);

  void distanceWorker();
  void computeAndPublishDistances(const rclcpp::Time &stamp,
                                  const std::vector<Box> &boxes);
  DistanceBundle computeDistanceBundle(const rclcpp::Time &stamp,
                                       const DistanceRequest &req);
  void publishDistanceBundle(const rclcpp::Time &stamp,
                             const DistanceBundle &bundle);
  void submitDistanceRequest(const rclcpp::Time &stamp,
                             const std::vector<Box> &boxes);
  bool tryGetDistanceCache(const rclcpp::Time &stamp,
                           DistanceBundle &out_bundle);

  int num_sources{4};                             // 输入源数量
  int process_frame_{0};                          // 处理帧计数
  std::vector<std::vector<Box>> previous_bboxes_; // 缓存上一次的检测结果

  // 线程相关
  std::thread detection_thread_;
  std::thread draw_thread_;
  std::thread distance_thread_;
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

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_pc2_sub_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr
      livox_custom_sub_;

  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::String>>
      detection_distance_pub_;

  std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>>
      bucket_bbox3d_marker_pub_;

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

  struct DrawItem
  {
    bool is_nv12{false};
    Nv12GpuFrame nv12_frame;
    cv::Mat bgr_frame;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
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

  DistanceEstimationConfig distance_cfg_;
  CamLidarCalib cam_lidar_calib_;

  std::mutex lidar_cloud_mutex_;
  LidarCloudCache lidar_cloud_cache_;

  struct LidarCloudFrame
  {
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    std::shared_ptr<const std::vector<LidarCamPoint>> points;
  };
  std::mutex lidar_cloud_history_mutex_;
  std::deque<LidarCloudFrame> lidar_cloud_history_;
  std::string last_lidar_frame_id_;

  struct Bbox3DConfig
  {
    bool enable{true};
    double window_ms{1000.0};
    int max_frames{20};
    int min_points{200};
    int max_points{200000};
    double max_age_ms{300.0};
    double depth_gate_m{1.2};
    double depth_bin_m{0.2};
    double roi_shrink_ratio{0.12};
    double hold_ms{500.0};
  };
  Bbox3DConfig bbox3d_cfg_;
  std::mutex bucket_bbox3d_cache_mutex_;
  std::vector<Bbox3DResult> bucket_bbox3d_cache_;
  rclcpp::Time bucket_bbox3d_cache_stamp_{0, 0, RCL_ROS_TIME};

  std::mutex distance_req_mutex_;
  std::condition_variable distance_req_cond_;
  DistanceRequest distance_req_;

  std::mutex distance_cache_mutex_;
  DistanceBundle distance_cache_;
  rclcpp::Time distance_cache_stamp_{0, 0, RCL_ROS_TIME};

  std::chrono::steady_clock::time_point fps_window_start_tp_{
      std::chrono::steady_clock::now()};
  uint32_t fps_window_frames_{0};
  double fps_value_{0.0};

public:
  // 构造/析构
  explicit CameraCombinedWithLidarPipelineManager(
      const InputConfig &input_config);
  CameraCombinedWithLidarPipelineManager();
  ~CameraCombinedWithLidarPipelineManager() override;

  // 核心接口
  void start() override;
  void stop() override;
  bool isRunning() const override;
};
