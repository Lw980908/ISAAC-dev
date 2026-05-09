#pragma once
#include "config.h"
#include <memory>
#include <rclcpp/rclcpp.hpp>

// 前向声明
class YOLOModel;
class TensorRTInference;
class MQTTPublish;
class UndistorterInterface;
class ROS2Client;

class PipelineInterface {
public:
  virtual ~PipelineInterface(); // 虚析构函数，在实现文件中定义
  // 配置参数
  ModelConfig model_config_;
  InputConfig input_config_;
  ToothDetectionConfig tooth_detection_config_;
  PedestrianDetectionConfig pedestrian_detection_config_;
  ImageProcessingConfig image_processing_config_;
  GeneralConfig general_config_;
  ROIConfig roi_config_;
  MqttConfig mqtt_config_;
  UndistortConfig undistort_config_;
  ImageEncoderDecoderConfig image_encoder_decoder_config_;

  // 核心接口
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool isRunning() const = 0;
  virtual void initResources() = 0;

  // 使用智能指针管理模型对象
  std::unique_ptr<YOLOModel> yolo_;
  std::unique_ptr<YOLOModel> second_yolo_;
  std::unique_ptr<TensorRTInference> trt_classifier_;
  std::unique_ptr<MQTTPublish> mqtt_publisher_;
  std::unique_ptr<UndistorterInterface> undistorter_;
  std::shared_ptr<ROS2Client> ros2_client_;

  rclcpp::Node::SharedPtr parent_node_ = nullptr; // 组件模式下使用的父节点
  // 设置父节点(组件模式使用)
  void setParentNode(rclcpp::Node::SharedPtr parent_node) {
    parent_node_ = parent_node;
  }

  // 配置设置方法
  void setROIConfig(const ROIConfig &config) { roi_config_ = config; }

  // GeneralConfig 包含 mutex 和 atomic，不能直接赋值，只复制需要的成员
  void setGeneralConfig(const GeneralConfig &config) {
    general_config_.is_show = config.is_show;
    general_config_.is_save = config.is_save;
    general_config_.show_gui = config.show_gui;
  }

  void setImageEncoderDecoderConfig(const ImageEncoderDecoderConfig &config) {
    image_encoder_decoder_config_ = config;
  }

  void setInputConfig(const InputConfig &config) { input_config_ = config; }

  void setModelConfig(const ModelConfig &config) { model_config_ = config; }
};