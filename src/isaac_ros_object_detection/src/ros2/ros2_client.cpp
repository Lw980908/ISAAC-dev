#include "ros2_client.h"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/logging.hpp>

// 构造函数1：创建自己的节点
ROS2Client::ROS2Client(InputConfig &input_config, const std::string &node_name,
                       size_t num_sources)
    : node_(std::make_shared<rclcpp::Node>(node_name)), owns_node_(true) {
  init(input_config, num_sources);
}

// 构造函数2：使用外部节点
ROS2Client::ROS2Client(rclcpp::Node::SharedPtr parent_node,
                       InputConfig &input_config, size_t num_sources)
    : node_(parent_node), owns_node_(false) {
  init(input_config, num_sources);
}

ROS2Client::~ROS2Client() {
  // 只在拥有节点所有权时才重置节点
  if (owns_node_) {
    node_.reset();
  }
}

void ROS2Client::init(InputConfig &input_config, size_t num_sources) {
  // 为每个相机创建独立的发布者
  detection_result_pubs_.resize(num_sources);

  for (size_t i = 0; i < num_sources; ++i) {
    std::string detect_topic = "/camera_" +
                               std::to_string(input_config.camera_ids[i]) +
                               "/detection_result";
    detection_result_pubs_[i] =
        node_->create_publisher<std_msgs::msg::String>(detect_topic, 1);
    RCLCPP_INFO(node_->get_logger(), "Source %zu topics: detect: %s", i,
                detect_topic.c_str());
  }
}

void ROS2Client::publishProcessedImage(const cv::Mat &img, size_t source_idx) {
  if (source_idx >= processed_image_pubs_.size() ||
      !processed_image_pubs_[source_idx]) {
    return;
  }

  // 将cv::Mat转换为sensor_msgs::msg::Image
  auto msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
  msg->header.stamp = node_->now();
  processed_image_pubs_[source_idx]->publish(*msg);
}

void ROS2Client::publishDetectionResult(const Box &box,
                                        const std::string &class_name,
                                        size_t source_idx) {
  if (source_idx >= detection_result_pubs_.size() ||
      !detection_result_pubs_[source_idx]) {
    return;
  }

  nlohmann::json obj;
  obj["class_name"] = class_name;
  obj["box"] = {static_cast<int>(box.left), static_cast<int>(box.top),
                static_cast<int>(box.right), static_cast<int>(box.bottom)};
  obj["width"] = static_cast<int>(box.right - box.left);
  obj["height"] = static_cast<int>(box.bottom - box.top);
  obj["score"] = box.confidence;
  obj["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();

  auto msg = std::make_shared<std_msgs::msg::String>();
  msg->data = obj.dump();
  detection_result_pubs_[source_idx]->publish(*msg); // 修复：添加publish调用
  RCLCPP_INFO(node_->get_logger(), "Published detection result: %s",
              msg->data.c_str());
}

void ROS2Client::publishDetectionResult(const std::string &class_name,
                                        size_t source_idx) {
  if (source_idx >= detection_result_pubs_.size() ||
      !detection_result_pubs_[source_idx]) {
    return;
  }

  nlohmann::json obj;
  obj["toothState"] = class_name;
  obj["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();

  auto msg = std::make_shared<std_msgs::msg::String>();
  msg->data = obj.dump();
  detection_result_pubs_[source_idx]->publish(*msg); // 修复：添加publish调用
  RCLCPP_INFO(node_->get_logger(), "Published detection result: %s",
              msg->data.c_str());
}

void ROS2Client::subscribeToOriginalImage(
    InputConfig &input_config, size_t source_idx,
    const std::function<
        void(const sensor_msgs::msg::CompressedImage::SharedPtr)> &callback) {
  std::string topic_name = "/camera_" +
                           std::to_string(input_config.camera_ids[source_idx]) +
                           "/compressed";
  auto subscription =
      node_->create_subscription<sensor_msgs::msg::CompressedImage>(
          topic_name, 10,
          [this,
           callback](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
            callback(msg);
          });

  original_image_subs_.push_back(subscription);
  original_image_callbacks_.push_back(callback);

  RCLCPP_INFO(node_->get_logger(), "Subscribed to original image topic: %s",
              topic_name.c_str());
}

void ROS2Client::subscribeToTopic(
    const std::string &topic_name,
    const std::function<void(const std_msgs::msg::String::SharedPtr)>
        &callback) {
  auto subscription = node_->create_subscription<std_msgs::msg::String>(
      topic_name, 10,
      [this, callback](const std_msgs::msg::String::SharedPtr msg) {
        callback(msg);
      });

  subscriptions_.push_back(subscription);
  subscription_callbacks_.push_back(callback);

  RCLCPP_INFO(node_->get_logger(), "Subscribed to topic: %s",
              topic_name.c_str());
}

// 新增：支持QoS和回调组的订阅方法
void ROS2Client::subscribeToOriginalImageWithQoS(
    InputConfig &input_config, size_t source_idx, const rclcpp::QoS &qos,
    rclcpp::CallbackGroup::SharedPtr group,
    const std::function<
        void(std::unique_ptr<sensor_msgs::msg::CompressedImage>)> &callback) {
  std::string topic_name = "/camera_" +
                           std::to_string(input_config.camera_ids[source_idx]) +
                           "/compressed";

  // 创建订阅选项并设置回调组
  rclcpp::SubscriptionOptions options;
  options.callback_group = group;

  // 创建订阅，使用MessageUniquePtr版本
  auto subscription =
      node_->create_subscription<sensor_msgs::msg::CompressedImage>(
          topic_name, qos,
          [callback](std::unique_ptr<sensor_msgs::msg::CompressedImage> msg) {
            // 直接将unique_ptr传递给用户回调
            callback(std::move(msg));
          },
          options);

  original_image_subs_.push_back(subscription);

  RCLCPP_INFO(
      node_->get_logger(),
      "Subscribed to original image topic with QoS and callback group: %s",
      topic_name.c_str());
}

void ROS2Client::subscribeToOriginalImageWithQoS(
    InputConfig &input_config, size_t source_idx, const rclcpp::QoS &qos,
    rclcpp::CallbackGroup::SharedPtr group,
    const std::function<void(std::unique_ptr<sensor_msgs::msg::Image>)>
        &callback) {
  std::string topic_name =
      "/camera_" + std::to_string(input_config.camera_ids[source_idx]) + "/raw";

  // 创建订阅选项并设置回调组
  rclcpp::SubscriptionOptions options;
  options.callback_group = group;

  // 创建订阅，使用MessageUniquePtr版本
  auto subscription = node_->create_subscription<sensor_msgs::msg::Image>(
      topic_name, qos,
      [callback](std::unique_ptr<sensor_msgs::msg::Image> msg) {
        // 直接将unique_ptr传递给用户回调
        callback(std::move(msg));
      },
      options);

  original_image_subs_yuyv_.push_back(subscription);

  RCLCPP_INFO(
      node_->get_logger(),
      "Subscribed to original image topic with QoS and callback group: %s",
      topic_name.c_str());
}