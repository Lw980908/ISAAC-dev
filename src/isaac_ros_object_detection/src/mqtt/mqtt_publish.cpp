#include "mqtt_publish.h"
#include <chrono>
#include <logger.h>

MQTTPublish::MQTTPublish(MqttConfig &mqtt_config)
    : MQTT_PUB_TOPIC(mqtt_config.MQTT_PUB_TOPIC),
      MQTT_BROKER(mqtt_config.MQTT_BROKER),
      MQTT_USERNAME(mqtt_config.MQTT_USERNAME),
      MQTT_PASSWORD(mqtt_config.MQTT_PASSWORD),
      MQTT_CLIENT_ID(mqtt_config.MQTT_CLIENT_ID),
      SUBSCRIBE_TOPIC(mqtt_config.subscribe_topic),
      SUBSCRIBE_FIELD(mqtt_config.subscribe_field),
      mqtt_enabled_(mqtt_config.mqtt_enabled) {

  if (!mqtt_enabled_) {
    return;
  }

  // 生成唯一的客户端ID
  auto now = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch())
                .count();
  MQTT_CLIENT_ID = mqtt_config.MQTT_CLIENT_ID + std::to_string(ns);
  sample::gLogInfo << "MQTT Client ID: " << MQTT_CLIENT_ID << std::endl;

  try {
    // 创建MQTT客户端
    mqtt_client_ = std::make_unique<MQTTClient>(
        MQTT_BROKER, MQTT_CLIENT_ID, SUBSCRIBE_TOPIC, SUBSCRIBE_FIELD,
        MQTT_USERNAME, MQTT_PASSWORD);

    // 设置检测控制回调
    mqtt_client_->set_detection_control_callback([this](bool enable) {
      subscribe_enabled_.store(enable, std::memory_order_release);
      sample::gLogInfo << "Detection " << (enable ? "enabled" : "disabled")
                       << " via MQTT" << std::endl;
    });

    // 异步连接
    mqtt_client_->connectAsync();

    // 异步订阅控制话题
    mqtt_client_->subscribeAsync(SUBSCRIBE_TOPIC, 1);
  } catch (const std::exception &e) {
    sample::gLogError << "MQTT Init Failed: " << e.what() << std::endl;
  }
}

MQTTPublish::~MQTTPublish() {
  if (mqtt_client_) {
    mqtt_client_->disconnectAsync();
  }
}

bool MQTTPublish::isMqttConnected() const {
  if (mqtt_client_)
    return mqtt_client_->is_connected();
  return false;
}

bool MQTTPublish::is_subscribe_enabled() const {
  return subscribe_enabled_.load(std::memory_order_acquire);
}

const std::string &MQTTPublish::getMqttTopic() const { return MQTT_PUB_TOPIC; }

bool MQTTPublish::publishAsync(const std::string &payload) {
  if (mqtt_client_ && isMqttConnected()) {
    return mqtt_client_->publishAsync(MQTT_PUB_TOPIC, payload, 0);
  }
  return false;
}