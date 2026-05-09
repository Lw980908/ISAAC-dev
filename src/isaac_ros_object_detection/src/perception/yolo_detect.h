#pragma once
#include "MobileNetV2/classify.h"
#include "config.h"
#include "distortion_correction/undistorter_interface.h"
#include "mqtt/mqtt_publish.h"
#include "ros2/ros2_client.h"
#include "yolo_model.h"

namespace yolo_detect {
void task(std::unique_ptr<YOLOModel> &yolo_,
          std::unique_ptr<YOLOModel> &second_yolo_,
          std::unique_ptr<TensorRTInference> &trt_classifier_,
          std::unique_ptr<MQTTPublish> &mqtt_publisher_,
          std::shared_ptr<ROS2Client> &ros2_client_, ModelConfig &model_config,
          ToothDetectionConfig &tooth_detection_config,
          PedestrianDetectionConfig &pedestrian_detection_config,
          ImageProcessingConfig &image_processing_config,
          GeneralConfig &general_config, InputConfig &input_config,
          ROIConfig &roi_config);

void detect(std::unique_ptr<YOLOModel> &yolo_,
            std::unique_ptr<YOLOModel> &second_yolo_,
            std::unique_ptr<TensorRTInference> &trt_classifier_,
            std::unique_ptr<UndistorterInterface> &undistorter_,
            std::unique_ptr<MQTTPublish> &mqtt_publisher_,
            std::shared_ptr<ROS2Client> &ros2_client_,
            ModelConfig &model_config, UndistortConfig &undistort_config,
            InputConfig &input_config,
            ToothDetectionConfig &tooth_detection_config,
            PedestrianDetectionConfig &pedestrian_detection_config,
            ImageProcessingConfig &image_processing_config,
            GeneralConfig &general_config, ROIConfig &roi_config,
            MqttConfig &mqtt_config);
} // namespace yolo_detect

template <typename T> T clamp(T value, T min_val, T max_val) {
  return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}
