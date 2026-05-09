#ifndef PERCEPTION_COMPONENT_H
#define PERCEPTION_COMPONENT_H

#include "thread_interface.h"
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace object_detection {
class PerceptionComponent : public rclcpp::Node {
public:
  explicit PerceptionComponent(const rclcpp::NodeOptions &options);
  ~PerceptionComponent();

  // 启动和停止pipeline的方法
  void startPipeline();
  void stopPipeline();

  // 添加定时器成员
  rclcpp::TimerBase::SharedPtr init_timer_;

private:
  std::unique_ptr<PipelineInterface> pipeline_manager_;
  InputConfig input_config_;
  ModelConfig model_config_;
  ToothDetectionConfig tooth_detection_config_;
  PedestrianDetectionConfig pedestrian_detection_config_;
  ImageProcessingConfig image_processing_config_;
  ROIConfig roi_config_;
  MqttConfig mqtt_config_;
  UndistortConfig undistort_config_;
  ImageEncoderDecoderConfig image_encoder_decoder_config_;
  GeneralConfig general_config_;
};

} // namespace object_detection

RCLCPP_COMPONENTS_REGISTER_NODE(object_detection::PerceptionComponent)

#endif // PERCEPTION_COMPONENT_H