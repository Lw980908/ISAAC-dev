#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mqtt/async_client.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <string>

class MQTTClient : public virtual mqtt::callback {
public:
  MQTTClient(const std::string &address, const std::string &client_id,
             const std::string &subscribe_topic,
             const std::string &subscribe_field, const std::string &username,
             const std::string &password);
  ~MQTTClient();

  bool connectAsync();
  void disconnectAsync();
  bool publishAsync(const std::string &topic, const std::string &payload,
                    int qos = 0);
  bool subscribeAsync(const std::string &topic, int qos = 1);

  void set_detection_control_callback(std::function<void(bool)> cb);

  bool is_connected() const;
  bool is_subscribe_enabled() const;

  // MQTT回调函数
  void connection_lost(const std::string &cause);
  void connected(const std::string &cause);
  void delivery_complete(mqtt::delivery_token_ptr token);
  void message_arrived(mqtt::const_message_ptr msg);

private:
  void workerThread();
  void processQueuedOperations();
  void processIncomingMessages();

  struct PublishItem {
    std::string topic;
    std::string payload;
    int qos;

    PublishItem(const std::string &t, const std::string &p, int q)
        : topic(t), payload(p), qos(q) {}
  };

  mqtt::async_client client_;
  std::string subscribe_topic_;
  std::string subscribe_field_;
  std::string username_;
  std::string password_;

  std::atomic<bool> connected_{false};
  std::atomic<bool> running_{true};
  std::atomic<bool> shutdown_requested_{false};
  std::atomic<bool> connect_requested_{false};
  std::atomic<bool> disconnect_requested_{false};
  std::atomic<bool> subscribe_enabled_{true};

  std::function<void(bool)> detection_control_callback_;

  // 队列用于线程间通信
  std::queue<PublishItem> publish_queue_;
  std::queue<std::string> subscribe_queue_;
  std::queue<mqtt::const_message_ptr> message_queue_;

  mutable std::mutex mtx_;
  std::condition_variable cv_;
  std::thread worker_thread_;
};