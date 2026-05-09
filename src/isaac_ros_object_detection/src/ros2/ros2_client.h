#ifndef ROS2_CLIENT_H
#define ROS2_CLIENT_H

#include "perception/config.h"
#include "rclcpp/callback_group.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_interfaces/node_base_interface.hpp"
#include "rclcpp/publisher.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/subscription.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

class ROS2Client {
public:
  // 构造函数1：创建自己的节点
  ROS2Client(InputConfig &input_config,
             const std::string &node_name = "ros2_client",
             size_t num_sources = 1);

  // 构造函数2：使用外部提供的节点
  ROS2Client(rclcpp::Node::SharedPtr parent_node, InputConfig &input_config,
             size_t num_sources = 1);

  ~ROS2Client();

  // 发布处理后的图像
  void publishProcessedImage(const cv::Mat &img, size_t source_idx);

  // 发布检测结果（JSON格式）
  void publishDetectionResult(const Box &box, const std::string &class_name,
                              size_t source_idx);
  void publishDetectionResult(const std::string &class_name, size_t source_idx);

  // 订阅原始图像话题
  void subscribeToOriginalImage(
      InputConfig &input_config, size_t source_idx,
      const std::function<
          void(const sensor_msgs::msg::CompressedImage::SharedPtr)> &callback);
  // 支持QoS和回调组的订阅方法 -- jpeg
  void subscribeToOriginalImageWithQoS(
      InputConfig &input_config, size_t source_idx, const rclcpp::QoS &qos,
      rclcpp::CallbackGroup::SharedPtr group,
      const std::function<
          void(std::unique_ptr<sensor_msgs::msg::CompressedImage>)> &callback);

  // 支持QoS和回调组的订阅方法 -- yuyv
  void subscribeToOriginalImageWithQoS(
      InputConfig &input_config, size_t source_idx, const rclcpp::QoS &qos,
      rclcpp::CallbackGroup::SharedPtr group,
      const std::function<void(std::unique_ptr<sensor_msgs::msg::Image>)>
          &callback);

  // 订阅其他模块的话题
  void subscribeToTopic(
      const std::string &topic_name,
      const std::function<void(const std_msgs::msg::String::SharedPtr)>
          &callback);

  // 获取ROS2节点指针
  rclcpp::Node::SharedPtr getNode() { return node_; }

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr
  get_node_base_interface() {
    return node_->get_node_base_interface();
  }

  void init(InputConfig &input_config, size_t num_sources);

private:
  rclcpp::Node::SharedPtr node_;
  bool owns_node_; // 标记是否拥有节点的所有权

  // 为每个输入源维护独立的发布者
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>
      processed_image_pubs_;
  std::vector<rclcpp::Publisher<std_msgs::msg::String>::SharedPtr>
      detection_result_pubs_;

  // 保存订阅的回调函数
  std::vector<
      std::function<void(const sensor_msgs::msg::CompressedImage::SharedPtr)>>
      original_image_callbacks_;
  std::vector<
      rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr>
      original_image_subs_;

  // std::vector<std::function<void(const sensor_msgs::msg::Image::SharedPtr)>>
  // original_image_callbacks_yuyv_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>
      original_image_subs_yuyv_;

  std::vector<std::function<void(const std_msgs::msg::String::SharedPtr)>>
      subscription_callbacks_;
  std::vector<rclcpp::Subscription<std_msgs::msg::String>::SharedPtr>
      subscriptions_;
};

#endif // ROS2_CLIENT_H
