#pragma once
#include "mqtt_client.h"
#include "perception/config.h"
#include <atomic>
#include <memory>

class MQTTPublish {
public:
  MQTTPublish(MqttConfig &mqtt_config);
  ~MQTTPublish();

  bool isMqttConnected() const;
  bool is_subscribe_enabled() const;
  const std::string &getMqttTopic() const;
  bool publishAsync(const std::string &payload);

private:
  std::string MQTT_PUB_TOPIC;
  std::string MQTT_BROKER;
  std::string MQTT_USERNAME;
  std::string MQTT_PASSWORD;
  std::string MQTT_CLIENT_ID;
  std::string SUBSCRIBE_TOPIC;
  std::string SUBSCRIBE_FIELD;

  std::unique_ptr<MQTTClient> mqtt_client_;
  std::atomic<bool> subscribe_enabled_{true};
  bool mqtt_enabled_;
};