#include "perception_component.h"
#include "pipeline_manager.h"
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>

using namespace object_detection;

// 辅助函数：将字符串转换为 EncodingFormat 枚举
static EncodingFormat parseEncodingFormat(const std::string &format_str) {
  if (format_str == "NITROS_NV12") {
    return EncodingFormat::NITROS_NV12;
  } else {
    return EncodingFormat::IMAGE_RAW; // 默认使用 sensor_msgs::msg::Image
  }
}

PerceptionComponent::PerceptionComponent(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options) {
  RCLCPP_INFO(this->get_logger(), "Initializing Perception Component");

  // 1. 从参数服务器加载配置
  this->declare_parameter(
      "process_method",
      static_cast<int>(ProcessMethod::SHARED_MODEL_THREAD_MULTI_SOURCES));
  this->declare_parameter("num_sources", 4);
  this->declare_parameter("ros2_enabled", true);
  this->declare_parameter("use_ros2_component_container", true);
  this->declare_parameter("show_gui", false);
  this->declare_parameter("camera_ids",
                          std::vector<int64_t>{0, 1, 2, 3}); // 改为int64_t

  // NITROS 格式配置参数
  this->declare_parameter("encoding_format_sub", std::string("IMAGE_RAW"));
  this->declare_parameter("encoding_format_pub", std::string("IMAGE_RAW"));
  this->declare_parameter(
      "camera_namespaces",
      std::vector<std::string>{"left", "right", "front", "back"});
  this->declare_parameter("nitros_topic_prefix", std::string(""));
  this->declare_parameter("nitros_pub_topic_prefix", std::string(""));

  // 改为一维数组，避免嵌套vector
  this->declare_parameter(
      "camera_topics",
      std::vector<std::string>{"/sensors/camera0/image/compressed",
                               "/sensors/camera1/image/compressed"});

  // 改为int64_t数组
  this->declare_parameter(
      "input_sizes",
      std::vector<int64_t>{1920, 1536, 1920, 1536, 1920, 1536, 1920, 1536});

  // 改为一维int64_t数组 (每4个值代表一个ROI: x, y, width, height)
  this->declare_parameter("roi_regions", std::vector<int64_t>{
                                             0, 0, 1920, 1536, // ROI 1
                                             0, 0, 1920, 1536, // ROI 2
                                             0, 0, 1920, 1536, // ROI 3
                                             0, 0, 1920, 1536  // ROI 4
                                         });

  // 2. 设置配置
  int process_method_int = this->get_parameter("process_method").as_int();
  input_config_.process_method = static_cast<ProcessMethod>(process_method_int);
  input_config_.ros2_enabled = this->get_parameter("ros2_enabled").as_bool();
  input_config_.use_ros2_component_container =
      this->get_parameter("use_ros2_component_container").as_bool();

  RCLCPP_INFO(this->get_logger(),
              "Perception params: process_method=%d ros2_enabled=%s "
              "use_ros2_component_container=%s",
              process_method_int, input_config_.ros2_enabled ? "true" : "false",
              input_config_.use_ros2_component_container ? "true" : "false");

  // 获取参数 - 使用正确的类型
  auto camera_ids = this->get_parameter("camera_ids")
                        .as_integer_array(); // 返回std::vector<int64_t>
  auto camera_topics = this->get_parameter("camera_topics").as_string_array();
  auto input_sizes = this->get_parameter("input_sizes")
                         .as_integer_array(); // 返回std::vector<int64_t>
  auto roi_regions = this->get_parameter("roi_regions")
                         .as_integer_array(); // 返回std::vector<int64_t>

  // NITROS 相关参数
  auto camera_namespaces =
      this->get_parameter("camera_namespaces").as_string_array();
  input_config_.camera_namespaces.assign(camera_namespaces.begin(),
                                         camera_namespaces.end());
  input_config_.nitros_topic_prefix =
      this->get_parameter("nitros_topic_prefix").as_string();
  input_config_.nitros_pub_topic_prefix =
      this->get_parameter("nitros_pub_topic_prefix").as_string();

  size_t num_sources = this->get_parameter("num_sources").as_int();

  // 如果从 camera_namespaces 获取到了源数量，使用它
  if (!camera_namespaces.empty()) {
    num_sources = camera_namespaces.size();
  }

  RCLCPP_INFO(this->get_logger(),
              "Perception sources: num_sources=%zu (camera_namespaces=%zu, "
              "camera_ids=%zu, input_sizes=%zu, roi_regions=%zu)",
              num_sources, camera_namespaces.size(), camera_ids.size(),
              input_sizes.size(), roi_regions.size());

  input_config_.camera_ids.resize(num_sources);
  input_config_.input_sizes.resize(num_sources);
  roi_config_.roi_regions.resize(num_sources);

  for (size_t i = 0; i < num_sources && i < camera_ids.size(); ++i) {
    input_config_.camera_ids[i] = static_cast<int>(camera_ids[i]); // 转换为int
  }

  for (size_t i = 0; i < num_sources && i * 2 + 1 < input_sizes.size(); ++i) {
    input_config_.input_sizes[i] =
        cv::Size(static_cast<int>(input_sizes[i * 2]),
                 static_cast<int>(input_sizes[i * 2 + 1]));
  }

  // 重构ROI区域
  for (size_t i = 0; i < num_sources && i * 4 < roi_regions.size(); ++i) {
    roi_config_.roi_regions[i] = {
        static_cast<int>(roi_regions[i * 4]),     // x
        static_cast<int>(roi_regions[i * 4 + 1]), // y
        static_cast<int>(roi_regions[i * 4 + 2]), // width
        static_cast<int>(roi_regions[i * 4 + 3])  // height
    };
  }

  // 3. NITROS 编码格式配置
  std::string encoding_sub_str =
      this->get_parameter("encoding_format_sub").as_string();
  std::string encoding_pub_str =
      this->get_parameter("encoding_format_pub").as_string();
  image_encoder_decoder_config_.encoding_format_sub =
      parseEncodingFormat(encoding_sub_str);
  image_encoder_decoder_config_.encoding_format_pub =
      parseEncodingFormat(encoding_pub_str);

  RCLCPP_INFO(this->get_logger(), "Encoding format - Sub: %s, Pub: %s",
              encoding_sub_str.c_str(), encoding_pub_str.c_str());
  RCLCPP_INFO(this->get_logger(), "Camera namespaces: %zu sources",
              camera_namespaces.size());

  // 5. GUI 配置
  general_config_.show_gui = this->get_parameter("show_gui").as_bool();

  // 6. 创建pipeline manager
  pipeline_manager_ = createPipelineManager(input_config_);

  // 7. 设置配置到 pipeline manager（需要在创建后设置）
  if (pipeline_manager_) {
    pipeline_manager_->setROIConfig(roi_config_);
    pipeline_manager_->setGeneralConfig(general_config_);
    pipeline_manager_->setImageEncoderDecoderConfig(
        image_encoder_decoder_config_);
  }

  // 8. 因生命安全周期问题，延迟启动pipeline
  init_timer_ =
      this->create_wall_timer(std::chrono::milliseconds(100), [this]() {
        this->init_timer_->cancel();
        // 设置父节点并启动
        pipeline_manager_->setParentNode(this->shared_from_this());
        startPipeline();
      });

  RCLCPP_INFO(this->get_logger(),
              "Perception component initialized successfully");
}

PerceptionComponent::~PerceptionComponent() {
  RCLCPP_INFO(this->get_logger(), "Shutting down Perception Component");
  stopPipeline();
  pipeline_manager_.reset();
}

void PerceptionComponent::startPipeline() {
  if (pipeline_manager_) {
    pipeline_manager_->initResources();
    pipeline_manager_->start();
  }
}

void PerceptionComponent::stopPipeline() {
  if (pipeline_manager_) {
    pipeline_manager_->stop();
  }
}
