#include "thread_interface.h"
#include "../MobileNetV2/classify.h"
#include "../distortion_correction/undistorter_interface.h"
#include "../mqtt/mqtt_publish.h"
#include "../ros2/ros2_client.h"
#include "../perception/yolo_model.h"

PipelineInterface::~PipelineInterface() {
  sample::gLogInfo << "PipelineInterface destructor called" << std::endl;
}
