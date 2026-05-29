#include "mqtt_client.h"
#include <chrono>
#include <logger.h>
#include <thread>

MQTTClient::MQTTClient(const std::string &address, const std::string &client_id,
                       const std::string &subscribe_topic,
                       const std::string &subscribe_field,
                       const std::string &username, const std::string &password)
    : client_(address, client_id), subscribe_topic_(subscribe_topic),
      subscribe_field_(subscribe_field), username_(username),
      password_(password) {
  // 启动工作线程
  worker_thread_ = std::thread(&MQTTClient::workerThread, this);
}

MQTTClient::~MQTTClient() {
  shutdown_requested_ = true;
  cv_.notify_one();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

bool MQTTClient::connectAsync() {
  std::lock_guard<std::mutex> lock(mtx_);
  // 如果已经连接或正在连接，不要重复请求
  if (connected_ || connect_requested_) {
    return false;
  }

  connect_requested_ = true;
  cv_.notify_one();
  return true;
}

void MQTTClient::disconnectAsync() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!connected_ && !connect_requested_) {
    return;
  }

  disconnect_requested_ = true;
  cv_.notify_one();
}

bool MQTTClient::publishAsync(const std::string &topic,
                              const std::string &payload, int qos) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!connected_) {
    return false;
  }

  publish_queue_.emplace(topic, payload, qos);
  cv_.notify_one();
  return true;
}

bool MQTTClient::subscribeAsync(const std::string &topic, int qos) {
  std::lock_guard<std::mutex> lock(mtx_);
  // 无论连接状态如何，都添加到订阅队列
  // 工作线程会确保只在连接状态下执行订阅
  subscribe_queue_.push(topic);
  cv_.notify_one();
  return true;
}

void MQTTClient::set_detection_control_callback(std::function<void(bool)> cb) {
  std::lock_guard<std::mutex> lock(mtx_);
  detection_control_callback_ = cb;
}

bool MQTTClient::is_connected() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return connected_;
}

bool MQTTClient::is_subscribe_enabled() const {
  return subscribe_enabled_.load(std::memory_order_acquire);
}

void MQTTClient::workerThread() {
  try {
    while (!shutdown_requested_) {
      std::unique_lock<std::mutex> lock(mtx_);

      // 检查是否有任务需要处理 - 添加了对消息队列的检查
      bool hasWork = !publish_queue_.empty() || !subscribe_queue_.empty() ||
                     connect_requested_ || disconnect_requested_ ||
                     !message_queue_.empty(); // 关键：添加消息队列检查

      if (!hasWork) {
        // 如果没有任务，等待条件变量
        cv_.wait_for(lock, std::chrono::milliseconds(100));
        continue;
      }

      // 优先处理断开请求
      if (disconnect_requested_) {
        disconnect_requested_ = false;
        lock.unlock();

        try {
          client_.disconnect()->wait_for(std::chrono::seconds(1));
        } catch (...) {
          // 忽略断开错误
        }

        std::lock_guard<std::mutex> lock(mtx_);
        connected_ = false;
        connect_requested_ = false; // 清除连接请求
        sample::gLogInfo << "MQTT Disconnected from broker" << std::endl;
        continue;
      }

      // 处理连接请求
      if (connect_requested_) {
        connect_requested_ = false;
        lock.unlock();

        try {
          auto connOpts = mqtt::connect_options_builder()
                              .user_name(username_)
                              .password(password_)
                              .clean_session(true)
                              .connect_timeout(std::chrono::seconds(5))
                              .finalize(); 

          client_.set_callback(*this);
          auto connToken = client_.connect(connOpts);

          // 等待连接结果
          if (connToken->wait_for(std::chrono::seconds(5)) &&
              connToken->is_complete()) {
            std::lock_guard<std::mutex> lock(mtx_);
            connected_ = true;
            sample::gLogInfo
                << "MQTT Connected to broker: " << client_.get_server_uri()
                << std::endl;
          } else {
            sample::gLogWarning << "MQTT connection timed out" << std::endl;
            // 稍后重试连接
            std::lock_guard<std::mutex> lock(mtx_);
            connect_requested_ = true;
            cv_.notify_one();
          }
        } catch (const mqtt::exception &e) {
          sample::gLogWarning << "MQTT Connect Error: " << e.what()
                              << std::endl;
          // 稍后重试连接
          std::this_thread::sleep_for(std::chrono::seconds(2));
          std::lock_guard<std::mutex> lock(mtx_);
          connect_requested_ = true;
          cv_.notify_one();
        }

        continue;
      }

      // 处理消息队列 - 关键修复：添加消息处理
      if (!message_queue_.empty()) {
        lock.unlock();
        processIncomingMessages();
        lock.lock();
      }

      // 只有在连接状态下才处理发布和订阅
      if (connected_) {
        lock.unlock();
        processQueuedOperations();
        lock.lock();
      }
    }
  } catch (const std::exception &e) {
    sample::gLogError << "MQTT Worker Thread Error: " << e.what() << std::endl;
  }
}

void MQTTClient::processQueuedOperations() {
  std::unique_lock<std::mutex> lock(mtx_);

  // 处理发布队列
  while (!publish_queue_.empty()) {
    auto item = publish_queue_.front();
    publish_queue_.pop();
    lock.unlock();

    try {
      auto msg = mqtt::make_message(item.topic, item.payload, item.qos, false);
      client_.publish(msg); // 异步发送
    } catch (const mqtt::exception &e) {
      sample::gLogError << "Publish Error: " << e.what() << std::endl;
      // 重新加入队列（但避免无限重试）
      std::lock_guard<std::mutex> lock(mtx_);
      if (publish_queue_.size() < 100) { // 限制队列大小
        publish_queue_.push(item);
      }
    }

    lock.lock();
  }

  // 处理订阅队列
  while (!subscribe_queue_.empty()) {
    auto topic = subscribe_queue_.front();
    subscribe_queue_.pop();
    lock.unlock();

    try {
      client_.subscribe(topic, 1); // 异步订阅
      sample::gLogInfo << "Subscribed to topic: " << topic << std::endl;
    } catch (const mqtt::exception &e) {
      sample::gLogError << "Subscribe Error: " << e.what() << std::endl;
      // 重新加入队列（但避免无限重试）
      std::lock_guard<std::mutex> lock(mtx_);
      if (subscribe_queue_.size() < 10) { // 限制队列大小
        subscribe_queue_.push(topic);
      }
    }

    lock.lock();
  }
}

void MQTTClient::processIncomingMessages() {
  std::unique_lock<std::mutex> lock(mtx_);

  while (!message_queue_.empty()) {
    auto msg = message_queue_.front();
    message_queue_.pop();
    lock.unlock();

    if (msg->get_topic() == subscribe_topic_) {
      try {
        nlohmann::json payload = nlohmann::json::parse(msg->to_string());
        if (payload.contains(subscribe_field_)) {
          int control_value = payload[subscribe_field_];
          sample::gLogInfo << "Detection control value received: "
                           << control_value << " (field: " << subscribe_field_
                           << ")" << std::endl;
          bool enable_detection = (control_value == 0);

          // 更新状态
          subscribe_enabled_.store(enable_detection, std::memory_order_release);

          // 调用回调
          if (detection_control_callback_) {
            detection_control_callback_(enable_detection);
          }
        } else if (payload.is_number()) {
          double value = payload.get<double>();
          sample::gLogInfo << "Value received: " << value << std::endl;
          bool enable_detection = (value == 0);

          // 更新状态
          subscribe_enabled_.store(enable_detection, std::memory_order_release);

          // 调用回调
          if (detection_control_callback_) {
            detection_control_callback_(enable_detection);
          }
        }
      } catch (const std::exception &e) {
        sample::gLogError << "Error parsing MQTT message: " << e.what()
                          << std::endl;
      }
    }

    lock.lock();
  }
}

void MQTTClient::connection_lost(const std::string &cause) {
  sample::gLogError << "MQTT Connection lost: " << cause << std::endl;

  std::lock_guard<std::mutex> lock(mtx_);
  connected_ = false;

  // 请求重新连接
  connect_requested_ = true;
  cv_.notify_one();
}

void MQTTClient::connected(const std::string &cause) {
  sample::gLogInfo << "MQTT Reconnected successfully: " << cause << std::endl;

  std::lock_guard<std::mutex> lock(mtx_);
  connected_ = true;

  // 唤醒工作线程处理待处理的订阅
  cv_.notify_one();
}

void MQTTClient::delivery_complete(mqtt::delivery_token_ptr token) {
  // 可选：实现消息确认处理
}

void MQTTClient::message_arrived(mqtt::const_message_ptr msg) {
  std::lock_guard<std::mutex> lock(mtx_);
  message_queue_.push(msg);
  cv_.notify_one(); // 通知工作线程处理消息
}