#include "camera_combined_with_lidar.h"

#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace
{
EncodingFormat ParseEncodingFormat(const std::string &format_str)
{
  if (format_str == "NITROS_NV12")
  {
    return EncodingFormat::NITROS_NV12;
  }
  return EncodingFormat::IMAGE_RAW;
}
}  // namespace

namespace isaac_ros_cam_and_lidar
{

class CameraCombinedWithLidarComponent : public rclcpp::Node
{
public:
  explicit CameraCombinedWithLidarComponent(
      const rclcpp::NodeOptions &options)
  : Node("camera_combined_with_lidar_component", options)
  {
    RCLCPP_INFO(
        this->get_logger(),
        "Initializing CameraCombinedWithLidar Component");

    this->declare_parameter("num_sources", 1);
    this->declare_parameter("ros2_enabled", true);
    this->declare_parameter("use_ros2_component_container", true);
    this->declare_parameter("show_gui", false);
    this->declare_parameter("camera_ids", std::vector<int64_t>{0});
    this->declare_parameter("encoding_format_sub", std::string("NITROS_NV12"));
    this->declare_parameter("encoding_format_pub", std::string("NITROS_NV12"));
    this->declare_parameter("camera_namespaces", std::vector<std::string>{"camera0"});
    this->declare_parameter("nitros_topic_prefix", std::string(""));
    this->declare_parameter("nitros_pub_topic_prefix", std::string(""));
    this->declare_parameter("input_sizes", std::vector<int64_t>{1920, 1536});
    this->declare_parameter("roi_regions", std::vector<int64_t>{0, 0, 1920, 1536});

    input_config_.ros2_enabled = this->get_parameter("ros2_enabled").as_bool();
    input_config_.use_ros2_component_container =
        this->get_parameter("use_ros2_component_container").as_bool();

    const auto camera_ids = this->get_parameter("camera_ids").as_integer_array();
    const auto input_sizes = this->get_parameter("input_sizes").as_integer_array();
    const auto roi_regions = this->get_parameter("roi_regions").as_integer_array();
    const auto camera_namespaces =
        this->get_parameter("camera_namespaces").as_string_array();

    input_config_.camera_namespaces.assign(
        camera_namespaces.begin(), camera_namespaces.end());
    input_config_.nitros_topic_prefix =
        this->get_parameter("nitros_topic_prefix").as_string();
    input_config_.nitros_pub_topic_prefix =
        this->get_parameter("nitros_pub_topic_prefix").as_string();

    size_t num_sources = static_cast<size_t>(
        this->get_parameter("num_sources").as_int());
    if (!camera_namespaces.empty())
    {
      num_sources = camera_namespaces.size();
    }

    input_config_.camera_ids.resize(num_sources);
    input_config_.input_sizes.resize(num_sources);
    roi_config_.roi_regions.resize(num_sources);

    for (size_t i = 0; i < num_sources && i < camera_ids.size(); ++i)
    {
      input_config_.camera_ids[i] = static_cast<int>(camera_ids[i]);
    }

    for (size_t i = 0; i < num_sources && i * 2 + 1 < input_sizes.size(); ++i)
    {
      input_config_.input_sizes[i] = cv::Size(
          static_cast<int>(input_sizes[i * 2]),
          static_cast<int>(input_sizes[i * 2 + 1]));
    }

    for (size_t i = 0; i < num_sources && i * 4 + 3 < roi_regions.size(); ++i)
    {
      roi_config_.roi_regions[i] = {
          static_cast<int>(roi_regions[i * 4]),
          static_cast<int>(roi_regions[i * 4 + 1]),
          static_cast<int>(roi_regions[i * 4 + 2]),
          static_cast<int>(roi_regions[i * 4 + 3])};
    }

    const std::string encoding_sub_str =
        this->get_parameter("encoding_format_sub").as_string();
    const std::string encoding_pub_str =
        this->get_parameter("encoding_format_pub").as_string();
    image_encoder_decoder_config_.encoding_format_sub =
        ParseEncodingFormat(encoding_sub_str);
    image_encoder_decoder_config_.encoding_format_pub =
        ParseEncodingFormat(encoding_pub_str);

    general_config_.show_gui = this->get_parameter("show_gui").as_bool();

    RCLCPP_INFO(
        this->get_logger(),
        "CameraCombinedWithLidar params: ros2_enabled=%s "
        "use_ros2_component_container=%s num_sources=%zu",
        input_config_.ros2_enabled ? "true" : "false",
        input_config_.use_ros2_component_container ? "true" : "false",
        num_sources);

    pipeline_manager_ =
        std::make_unique<CameraCombinedWithLidarPipelineManager>(input_config_);
    pipeline_manager_->setROIConfig(roi_config_);
    pipeline_manager_->setGeneralConfig(general_config_);
    pipeline_manager_->setImageEncoderDecoderConfig(
        image_encoder_decoder_config_);

    init_timer_ =
        this->create_wall_timer(std::chrono::milliseconds(100), [this]()
                                {
      this->init_timer_->cancel();
      pipeline_manager_->setParentNode(this->shared_from_this());
      pipeline_manager_->initResources();
      pipeline_manager_->start(); });

    RCLCPP_INFO(
        this->get_logger(),
        "CameraCombinedWithLidar component initialized successfully");
  }

  ~CameraCombinedWithLidarComponent() override
  {
    RCLCPP_INFO(
        this->get_logger(),
        "Shutting down CameraCombinedWithLidar Component");
    if (pipeline_manager_)
    {
      pipeline_manager_->stop();
      pipeline_manager_.reset();
    }
  }

private:
  rclcpp::TimerBase::SharedPtr init_timer_;
  std::unique_ptr<CameraCombinedWithLidarPipelineManager> pipeline_manager_;
  InputConfig input_config_;
  ROIConfig roi_config_;
  ImageEncoderDecoderConfig image_encoder_decoder_config_;
  GeneralConfig general_config_;
};

}  // namespace isaac_ros_cam_and_lidar

RCLCPP_COMPONENTS_REGISTER_NODE(
    isaac_ros_cam_and_lidar::CameraCombinedWithLidarComponent)
