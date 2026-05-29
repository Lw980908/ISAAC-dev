#include "yolo_detect.h"
#include "distortion_correction/undistort_factory.h"
#include <boost/filesystem.hpp>
#include <future>
#include <nlohmann/json.hpp>
#include <thread>

struct BoxComparator
{
  bool operator()(const Box &a, const Box &b) const
  {
    return a.confidence < b.confidence;
  }
};

namespace
{
  void waitAllTasks(std::vector<std::future<void>> &tasks)
  {
    for (auto &task : tasks)
    {
      if (task.valid())
      {
        task.wait();
      }
    }
    tasks.clear();
  }

  bool handlePeopleBox(Box &box, size_t batch_index,
                       std::unique_ptr<TensorRTInference> &trt_classifier_,
                       ModelConfig &model_config,
                       PedestrianDetectionConfig &pedestrian_detection_config,
                       ImageProcessingConfig &image_processing_config,
                       InputConfig &input_config,
                       std::shared_ptr<ROS2Client> &ros2_client_,
                       std::mutex &classifier_mtx,
                       std::vector<std::future<void>> &classification_tasks)
  {
    const bool do_second_classify = model_config.need_second_detection_ &&
                                    model_config.need_classifier_ &&
                                    trt_classifier_ && trt_classifier_->isReady();

    if (!do_second_classify)
    {
      if (box.confidence > 0.7f)
      {
        ++pedestrian_detection_config.people_cnt;
        if (input_config.ros2_enabled && ros2_client_)
        {
          ros2_client_->publishDetectionResult(box, "people", input_config.idx);
        }
        return true;
      }
      if (box.confidence < 0.3f)
      {
        box.confidence = 0.0f;
        return true;
      }

      if (!model_config.need_classifier_)
      {
        ++pedestrian_detection_config.people_cnt;
        if (input_config.ros2_enabled && ros2_client_)
        {
          ros2_client_->publishDetectionResult(box, "people", input_config.idx);
        }
        return true;
      }
    }

    int x = static_cast<int>(box.left);
    int y = static_cast<int>(box.top);
    int width = static_cast<int>(box.right - box.left);
    int height = static_cast<int>(box.bottom - box.top);

    x = std::max(
        0, std::min(x, image_processing_config.crop_batch[batch_index].cols - 1));
    y = std::max(
        0, std::min(y, image_processing_config.crop_batch[batch_index].rows - 1));
    width = std::max(
        1, std::min(width,
                    image_processing_config.crop_batch[batch_index].cols - x));
    height = std::max(
        1, std::min(height,
                    image_processing_config.crop_batch[batch_index].rows - y));

    cv::Rect roi(x, y, width, height);
    cv::Mat cropped_image = image_processing_config.crop_batch[batch_index](roi);

    if (classification_tasks.size() < model_config.classifier_thread_limit)
    {
      auto future = model_config.classifier_thread_pool_->enqueue(
          [&, cropped_image, box_ptr = &box]()
          {
            auto classif_start = std::chrono::steady_clock::now();
            std::vector<float> classifier_output;
            {
              std::lock_guard<std::mutex> lock(classifier_mtx);
              trt_classifier_->predict(cropped_image, classifier_output);
            }
            auto classif_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> detect_duration =
                classif_end - classif_start;
            sample::gLogInfo << "Classification time: " << detect_duration.count()
                             << " ms" << std::endl;
            float person_confidence = classifier_output[0];
            if (person_confidence > 0.5f)
            {
              __atomic_fetch_add(&pedestrian_detection_config.people_cnt, 1,
                                 __ATOMIC_RELAXED);
              if (input_config.ros2_enabled && ros2_client_)
              {
                ros2_client_->publishDetectionResult(*box_ptr, "people",
                                                     input_config.idx);
              }
            }
            else
            {
              box_ptr->confidence = 0.0f;
            }
          });
      classification_tasks.push_back(std::move(future));
      return true;
    }

    auto classif_start = std::chrono::steady_clock::now();
    std::vector<float> classifier_output;
    trt_classifier_->predict(cropped_image, classifier_output);
    auto classif_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> detect_duration = classif_end - classif_start;
    sample::gLogInfo << "Classification time: " << detect_duration.count()
                     << " ms" << std::endl;
    float person_confidence = classifier_output[0];

    if (person_confidence > 0.5f)
    {
      ++pedestrian_detection_config.people_cnt;
      if (input_config.ros2_enabled && ros2_client_)
      {
        ros2_client_->publishDetectionResult(box, "people", input_config.idx);
      }
    }
    else
    {
      box.confidence = 0.0f;
    }
    return true;
  }

  void runSecondBucketDetection(std::unique_ptr<YOLOModel> &second_yolo_,
                                size_t batch_index, const Box &bucket_box,
                                ToothDetectionConfig &tooth_detection_config,
                                ImageProcessingConfig &image_processing_config,
                                std::mutex &bucket_mtx)
  {
    int x = static_cast<int>(bucket_box.left);
    int y = static_cast<int>(bucket_box.top);
    int width = static_cast<int>(bucket_box.right - bucket_box.left);
    int height = static_cast<int>(bucket_box.bottom - bucket_box.top);

    x = clamp(x, 0, image_processing_config.crop_batch[batch_index].cols - 1);
    y = clamp(y, 0, image_processing_config.crop_batch[batch_index].rows - 1);
    width =
        clamp(width, 0, image_processing_config.crop_batch[batch_index].cols - x);
    height = clamp(height, 0,
                   image_processing_config.crop_batch[batch_index].rows - y);

    cv::Rect roi(x, y, width, height);
    cv::Mat cropped_image = image_processing_config.crop_batch[batch_index](roi);

    int original_width = cropped_image.cols;
    int original_height = cropped_image.rows;
    cv::Size target_size(640, 640);
    cv::resize(cropped_image, cropped_image, target_size);

    float scale_x = static_cast<float>(target_size.width) / original_width;
    float scale_y = static_cast<float>(target_size.height) / original_height;

    std::vector<cv::Mat> crop_batch = {cropped_image};
    second_yolo_->getYolo().copy(crop_batch);
    second_yolo_->getYolo().preprocess(crop_batch);
    second_yolo_->getYolo().infer();
    second_yolo_->getYolo().postprocess(crop_batch);
    auto teeth_boxes = second_yolo_->getYolo().getObjectss()[0];
    second_yolo_->getYolo().reset();

    for (auto &teeth_box : teeth_boxes)
    {
      if (!tooth_detection_config.second_detection_labels.count(
              second_yolo_->getParam().class_names[teeth_box.label]))
      {
        continue;
      }
      teeth_box.left /= scale_x;
      teeth_box.right /= scale_x;
      teeth_box.top /= scale_y;
      teeth_box.bottom /= scale_y;

      float left =
          std::max(0.0f, std::min(teeth_box.left,
                                  static_cast<float>(cropped_image.cols - 1)));
      float right =
          std::max(0.0f, std::min(teeth_box.right,
                                  static_cast<float>(cropped_image.cols - 1)));
      float top =
          std::max(0.0f, std::min(teeth_box.top,
                                  static_cast<float>(cropped_image.rows - 1)));
      float bottom =
          std::max(0.0f, std::min(teeth_box.bottom,
                                  static_cast<float>(cropped_image.rows - 1)));

      teeth_box.left = left + x;
      teeth_box.right = right + x;
      teeth_box.top = top + y;
      teeth_box.bottom = bottom + y;

      teeth_box.left = std::max(
          0, std::min(static_cast<int>(teeth_box.left),
                      image_processing_config.crop_batch[batch_index].cols - 1));
      teeth_box.right = std::max(
          0, std::min(static_cast<int>(teeth_box.right),
                      image_processing_config.crop_batch[batch_index].cols - 1));
      teeth_box.top = std::max(
          0, std::min(static_cast<int>(teeth_box.top),
                      image_processing_config.crop_batch[batch_index].rows - 1));
      teeth_box.bottom = std::max(
          0, std::min(static_cast<int>(teeth_box.bottom),
                      image_processing_config.crop_batch[batch_index].rows - 1));

      std::lock_guard<std::mutex> lock(bucket_mtx);
      image_processing_config.bboxes[batch_index].push_back(teeth_box);
      if (tooth_detection_config.tooth_state.count(teeth_box.label))
      {
        tooth_detection_config.tooth_state_name.emplace(
            second_yolo_->getParam().class_names[teeth_box.label]);
        // 正铲挖掘机缺齿时无齿根，插入状态信息作为哨兵
        if (tooth_detection_config.excavator_type ==
            ExcavatorType::Front_shovel_excavator)
        {
          tooth_detection_config.teeth_root_coordinates.emplace_back(
              std::make_tuple(teeth_box.label, (int)(teeth_box.left - 50),
                              (int)(teeth_box.left - 50)));
          tooth_detection_config.teeth_root_coordinates.emplace_back(
              std::make_tuple(teeth_box.label, (int)(teeth_box.right + 50),
                              (int)(teeth_box.right + 50)));
        }
      }
      else if (tooth_detection_config.teeth_and_root.count(teeth_box.label))
      {
        tooth_detection_config.teeth_root_coordinates.emplace_back(
            std::make_tuple(teeth_box.label, (int)(teeth_box.left),
                            (int)(teeth_box.right)));
      }
    }
  }

  void handleSecondDetectionBox(
      Box &box, size_t batch_index, std::unique_ptr<YOLOModel> &yolo_,
      std::unique_ptr<YOLOModel> &second_yolo_, ModelConfig &model_config,
      ToothDetectionConfig &tooth_detection_config,
      ImageProcessingConfig &image_processing_config,
      std::priority_queue<Box, std::vector<Box>, BoxComparator> &bucket_pq,
      std::vector<std::future<void>> &bucket_tasks, std::mutex &bucket_mtx)
  {
    if (box.label < yolo_->getParam().class_names.size() &&
        yolo_->getParam().class_names[box.label] == "reset")
    {
      bucket_pq.emplace(box);
    }
    else
    { // 清除第一次目标检测时的其它类别信息（斗齿等），只保留铲斗信息用作二次检测
      box.confidence = 0.0f;
    }

    if (bucket_pq.empty())
    {
      return;
    }

    auto bucket_box = bucket_pq.top();
    bucket_pq.pop();

    auto future =
        model_config.bucket_thread_pool_->enqueue([&, batch_index, bucket_box]()
                                                  { runSecondBucketDetection(second_yolo_, batch_index, bucket_box,
                                                                             tooth_detection_config,
                                                                             image_processing_config, bucket_mtx); });
    bucket_tasks.push_back(std::move(future));
  }
} // namespace

void yolo_detect::task(std::unique_ptr<YOLOModel> &yolo_,
                       std::unique_ptr<YOLOModel> &second_yolo_,
                       std::unique_ptr<TensorRTInference> &trt_classifier_,
                       std::unique_ptr<MQTTPublish> &mqtt_publisher_,
                       std::shared_ptr<ROS2Client> &ros2_client_,
                       ModelConfig &model_config,
                       ToothDetectionConfig &tooth_detection_config,
                       PedestrianDetectionConfig &pedestrian_detection_config,
                       ImageProcessingConfig &image_processing_config,
                       GeneralConfig &general_config, InputConfig &input_config,
                       ROIConfig &roi_config)
{
  image_processing_config.bboxes.clear(); // 清除上次的检测结果

  if (mqtt_publisher_ && !mqtt_publisher_->is_subscribe_enabled())
  {
    sample::gLogInfo << "Detection is disabled, clearing previous alerts and "
                        "skipping processing"
                     << std::endl;

    // 清除行人报警状态
    if (pedestrian_detection_config.is_people_detected &&
        mqtt_publisher_->isMqttConnected())
    {
      nlohmann::json payload_json;
      payload_json["peopleCount"] = 0;
      std::string payload = payload_json.dump();

      if (!mqtt_publisher_->publishAsync(payload))
      {
        sample::gLogWarning << "Failed to publish peopleCount reset message"
                            << std::endl;
      }
      else
      {
        sample::gLogInfo << "Published peopleCount reset: " << payload
                         << std::endl;
      }

      // 重置行人检测状态
      pedestrian_detection_config.last_people_detected_time_ =
          std::chrono::steady_clock::time_point();
      pedestrian_detection_config.is_people_detected = false;
      pedestrian_detection_config.show_people_cnt = 0;
    }

    // 清除斗齿缺失报警状态
    if (tooth_detection_config.is_tooth_detected &&
        mqtt_publisher_->isMqttConnected())
    {
      nlohmann::json payload_json;
      payload_json["toothDetection"] =
          tooth_detection_config.detection_mapping["complete"];
      std::string payload = payload_json.dump();

      if (!mqtt_publisher_->publishAsync(payload))
      {
        sample::gLogWarning << "Failed to publish toothDetection reset message"
                            << std::endl;
      }
      else
      {
        sample::gLogInfo << "Published toothDetection reset: " << payload
                         << std::endl;
      }

      // 重置斗齿检测状态
      tooth_detection_config.last_tooth_detected_time_ =
          std::chrono::steady_clock::time_point();
      tooth_detection_config.is_tooth_detected = false;
      tooth_detection_config.show_tooth_state_name = "complete";
    }

    // 设置显示标志
    image_processing_config.need_show_detection_info_ = false;
    image_processing_config.need_draw_detection_area_ = false;

    // 清除显示区域的检测结果
    image_processing_config.bboxes.clear();
    pedestrian_detection_config.people_cnt = 0;
    tooth_detection_config.tooth_state_name.clear();
    tooth_detection_config.teeth_root_coordinates.clear();
    tooth_detection_config.root_idx.clear();
    tooth_detection_config.match_cnt = 0;
    tooth_detection_config.max_count = 0;
    tooth_detection_config.max_class_label = "";
    tooth_detection_config.detect_res.clear();

    return;
  }
  // else
  // {
  //     image_processing_config.need_show_detection_info_ = true;
  //     image_processing_config.need_draw_detection_area_ = true;
  // }
  auto start = std::chrono::steady_clock::now();
  // yolo_utils::DeviceTimer d_t0;
  yolo_->getYolo().copy(image_processing_config.crop_batch);
  // float t0 = d_t0.getUsedTime();

  // yolo_utils::DeviceTimer d_t1;
  yolo_->getYolo().preprocess(image_processing_config.crop_batch);
  // float t1 = d_t1.getUsedTime();

  // yolo_utils::DeviceTimer d_t2;
  yolo_->getYolo().infer();
  // float t2 = d_t2.getUsedTime();

  // yolo_utils::DeviceTimer d_t3;
  yolo_->getYolo().postprocess(image_processing_config.crop_batch);
  // float t3 = d_t3.getUsedTime();
  // sample::gLogInfo << "preprocess time = " << t1 /
  // yolo_->getParam().batch_size << "; "
  //                  << "infer time = " << t2 / yolo_->getParam().batch_size <<
  //                  "; "
  //                  << "postprocess time = " << t3 /
  //                  yolo_->getParam().batch_size << std::endl;

  // 获取一个批次（预设为4）的推理信息
  image_processing_config.bboxes = std::move(yolo_->getYolo().getObjectss());
  // int sz1 = image_processing_config.bboxes[0].size();
  // std::cout << sz1 << std::endl;

  // 清除本次的推理结果
  yolo_->getYolo().reset();

  std::mutex classifier_mtx;
  std::mutex bucket_mtx;
  // 遍历每批图像中的图片
  for (size_t i = 0; i < image_processing_config.bboxes.size(); ++i)
  {
    std::vector<std::future<void>> classification_tasks;
    std::vector<std::future<void>> bucket_tasks;
    std::priority_queue<Box, std::vector<Box>, BoxComparator> bucket_pq;
    for (auto &box :
         image_processing_config.bboxes[i]) // 遍历每张图片中的检测框
    {
      // 行人检测
      if (yolo_->getParam().class_names[box.label] == "people")
      {
        if (handlePeopleBox(box, i, trt_classifier_, model_config,
                            pedestrian_detection_config,
                            image_processing_config, input_config, ros2_client_,
                            classifier_mtx, classification_tasks))
        {
          continue;
        }
      }

      // ===========================================
      // 斗齿检测
      if (model_config.need_second_detection_)
      {
        handleSecondDetectionBox(
            box, i, yolo_, second_yolo_, model_config, tooth_detection_config,
            image_processing_config, bucket_pq, bucket_tasks, bucket_mtx);
      }
      else
      {
        if (tooth_detection_config.tooth_state.count(
                box.label)) // 铲斗状态信息（完整或缺失）
        {
          tooth_detection_config.tooth_state_name.emplace(
              yolo_->getParam().class_names[box.label]);
          // 正铲挖掘机缺齿时无齿根，插入状态信息作为哨兵
          if (tooth_detection_config.excavator_type ==
              ExcavatorType::Front_shovel_excavator)
          {
            tooth_detection_config.teeth_root_coordinates.emplace_back(
                std::make_tuple(box.label, (int)(box.left - 50),
                                (int)(box.left - 50)));
            tooth_detection_config.teeth_root_coordinates.emplace_back(
                std::make_tuple(box.label, (int)(box.right + 50),
                                (int)(box.right + 50)));
          }
        }
        else if (tooth_detection_config.teeth_and_root.count(
                     box.label)) // 斗齿信息
        {
          tooth_detection_config.teeth_root_coordinates.emplace_back(
              std::make_tuple(box.label, (int)(box.left), (int)(box.right)));
        }
      }
    }

    // 等待所有异步分类完成
    if (model_config.need_classifier_)
    {
      waitAllTasks(classification_tasks);
    }

    // 等待所有异步斗齿检测完成
    if (model_config.need_second_detection_)
    {
      waitAllTasks(bucket_tasks);
    }

    // 移除无效框
    if (model_config.need_classifier_ || model_config.need_second_detection_)
    {
      image_processing_config.bboxes[i].erase(
          std::remove_if(image_processing_config.bboxes[i].begin(),
                         image_processing_config.bboxes[i].end(),
                         [](const Box &b)
                         { return b.confidence == 0.0f; }),
          image_processing_config.bboxes[i].end());
    }

    // int sz2 = image_processing_config.bboxes[0].size();
    // if (sz1 != sz2)
    // {
    //     std::cout << sz1 << " " << sz2 << " " <<
    //     pedestrian_detection_config.people_cnt << std::endl;
    // }

    pedestrian_detection_config.show_people_cnt =
        pedestrian_detection_config.people_cnt;
    if (pedestrian_detection_config.show_people_cnt &&
        mqtt_publisher_->isMqttConnected())
    {
      // 创建 JSON 对象
      nlohmann::json payload_json;
      payload_json["peopleCount"] = pedestrian_detection_config.show_people_cnt;
      // 将 JSON 对象序列化为字符串
      std::string payload = payload_json.dump();

      // 使用异步发布（不再需要锁）
      if (!mqtt_publisher_->publishAsync(payload))
      {
        sample::gLogWarning << "MQTT Publish Failed!" << std::endl;
      }
      sample::gLogInfo << "MQTT Publish Success! Payload: " << payload
                       << std::endl;

      // 更新最后检测时间
      pedestrian_detection_config.last_people_detected_time_ =
          std::chrono::steady_clock::now();
      pedestrian_detection_config.is_people_detected = true;
    }

    // 未检测到行人超过1秒，解除大屏报警
    if (pedestrian_detection_config.is_people_detected)
    {
      auto current_time = std::chrono::steady_clock::now();
      if (pedestrian_detection_config.last_people_detected_time_ !=
              std::chrono::steady_clock::time_point() &&
          (current_time -
           pedestrian_detection_config.last_people_detected_time_) >
              std::chrono::seconds(1))
      {
        if (pedestrian_detection_config.show_people_cnt == 0 &&
            mqtt_publisher_->isMqttConnected())
        {
          nlohmann::json payload_json;
          payload_json["peopleCount"] = 0;
          std::string payload = payload_json.dump();

          // 使用异步发布（不再需要锁）
          if (!mqtt_publisher_->publishAsync(payload))
          {
            sample::gLogWarning << "MQTT Publish Failed!" << std::endl;
          }
          sample::gLogInfo << "MQTT Publish Success! Payload: " << payload
                           << std::endl;

          // 重置时间戳，避免重复发送0
          pedestrian_detection_config.last_people_detected_time_ =
              std::chrono::steady_clock::time_point();
          pedestrian_detection_config.is_people_detected = false;
        }
      }
    }

    // 铲斗状态信息必须只能是一个，否则说明存在误检
    if (tooth_detection_config.tooth_state_name.size() == 1)
    {
      if (tooth_detection_config.excavator_type ==
          ExcavatorType::Back_shovel_excavator) // 反铲挖掘机缺齿时有齿根
      {
        if (tooth_detection_config.teeth_root_coordinates.size() ==
            tooth_detection_config
                .teeth_complete_number) // 齿尖和齿根要全部检测全（即斗齿完整时的齿尖数量），防止漏检
        {
          std::sort(tooth_detection_config.teeth_root_coordinates.begin(),
                    tooth_detection_config.teeth_root_coordinates.end(),
                    [&](const std::tuple<int, int, int> &a,
                        const std::tuple<int, int, int> &b)
                    {
                      return std::get<1>(a) <
                             std::get<1>(
                                 b); // 获取元组中的第二个元素 (box.left)
                    });              // 对 teeth_root_coordinates 按照 box.left 升序排序
          // 获取齿根所在位置
          for (int i = 0;
               i < tooth_detection_config.teeth_root_coordinates.size(); ++i)
          {
            if (yolo_->getParam().class_names[std::get<0>(
                    tooth_detection_config.teeth_root_coordinates[i])] ==
                "root")
            {
              tooth_detection_config.root_idx.emplace_back(i + 1);
              // std::cout << "缺失位置：" << i + 1 << std::endl;
            }
          }

          // 联合匹配
          if (tooth_detection_config.root_idx.empty() &&
              *tooth_detection_config.tooth_state_name.begin() == "complete")
          {
            ++tooth_detection_config.detect_res["complete"];
            ++tooth_detection_config.match_cnt;
            if (tooth_detection_config.detect_res["complete"] >
                tooth_detection_config.max_count)
            {
              tooth_detection_config.max_count =
                  tooth_detection_config.detect_res["complete"];
              tooth_detection_config.max_class_label = "complete";
            }
          }
          else
          {
            for (auto &idx :
                 tooth_detection_config.root_idx) // 是否缺失多个斗齿
            {
              std::string label = "miss" + std::to_string(idx);
              if (*tooth_detection_config.tooth_state_name.begin() == label)
              {
                ++tooth_detection_config.detect_res[label];
                ++tooth_detection_config.match_cnt;
                if (tooth_detection_config.detect_res[label] >
                    tooth_detection_config.max_count)
                {
                  tooth_detection_config.max_count =
                      tooth_detection_config.detect_res[label];
                  tooth_detection_config.max_class_label = label;
                }
              }
            }
          }
        }
      }
      else // 正铲挖掘机缺齿时无齿根
      {
        // 斗齿完整
        if (tooth_detection_config.teeth_root_coordinates.size() ==
                tooth_detection_config.teeth_complete_number + 2 &&
            *tooth_detection_config.tooth_state_name.begin() == "complete")
        {
          ++tooth_detection_config.detect_res["complete"];
          ++tooth_detection_config.match_cnt;
          if (tooth_detection_config.detect_res["complete"] >
              tooth_detection_config.max_count)
          {
            tooth_detection_config.max_count =
                tooth_detection_config.detect_res["complete"];
            tooth_detection_config.max_class_label = "complete";
          }
        }
        else
        { // 缺齿
          std::sort(tooth_detection_config.teeth_root_coordinates.begin(),
                    tooth_detection_config.teeth_root_coordinates.end(),
                    [&](const std::tuple<int, int, int> &a,
                        const std::tuple<int, int, int> &b)
                    {
                      return std::get<1>(a) < std::get<1>(b);
                    }); // 对 teeth_root_coordinates 按照 box.left 升序排序
          for (int i = 0;
               i < tooth_detection_config.teeth_root_coordinates.size() - 1;
               ++i)
          {
            int dist = abs(
                std::get<1>(
                    tooth_detection_config.teeth_root_coordinates[i + 1]) -
                std::get<1>(tooth_detection_config.teeth_root_coordinates[i]));
            tooth_detection_config.root_idx.emplace_back(dist);
            // std::cout << dist << std::endl;
          }
          auto max_gap_idx =
              std::max_element(tooth_detection_config.root_idx.begin(),
                               tooth_detection_config.root_idx.end()) -
              tooth_detection_config.root_idx.begin();
          std::string label = "miss" + std::to_string(max_gap_idx + 1);
          // std::cout << label << std::endl;
          if (*tooth_detection_config.tooth_state_name.begin() == label)
          {
            ++tooth_detection_config.detect_res[label];
            ++tooth_detection_config.match_cnt;
            if (tooth_detection_config.detect_res[label] >
                tooth_detection_config.max_count)
            {
              tooth_detection_config.max_count =
                  tooth_detection_config.detect_res[label];
              tooth_detection_config.max_class_label = label;
            }
          }
        }
      }
      // std::cout << tooth_detection_config.max_count << " " <<
      // tooth_detection_config.max_class_label << std::endl; std::cout <<
      // "当前匹配计数: " << tooth_detection_config.match_cnt << ", 最大计数: "
      // << tooth_detection_config.max_count << ", 状态: " <<
      // tooth_detection_config.max_class_label << std::endl;
      if (tooth_detection_config.match_cnt % 5 == 0 &&
          tooth_detection_config.max_count >
              tooth_detection_config.match_cnt * 0.7)
      {
        tooth_detection_config.show_tooth_state_name =
            tooth_detection_config.max_class_label;
        sample::gLogInfo << "斗齿监测状态为："
                         << tooth_detection_config.max_class_label << "，每 "
                         << tooth_detection_config.match_cnt
                         << " 次匹配，检测到该状态的次数为："
                         << tooth_detection_config.max_count << std::endl;

        if (tooth_detection_config.max_class_label != "complete")
        {
          if (mqtt_publisher_->isMqttConnected())
          { // 创建 JSON 对象
            nlohmann::json payload_json;
            payload_json["toothDetection"] =
                tooth_detection_config
                    .detection_mapping[tooth_detection_config.max_class_label];
            // 将 JSON 对象序列化为字符串
            std::string payload = payload_json.dump();
            // 使用异步发布（不再需要锁）
            if (!mqtt_publisher_->publishAsync(payload))
            {
              sample::gLogWarning << "MQTT Publish Failed!" << std::endl;
            }
            sample::gLogInfo << "MQTT Publish Success! Payload: " << payload
                             << std::endl;

            // 更新最后检测时间
            tooth_detection_config.last_tooth_detected_time_ =
                std::chrono::steady_clock::now();
            tooth_detection_config.is_tooth_detected = true;
          }

          if (input_config.ros2_enabled && ros2_client_)
          {
            ros2_client_->publishDetectionResult(
                tooth_detection_config.max_class_label, input_config.idx);
          }
        }

        // if (input_config.source == InputStream::VP_filter)
        // {
        //     sample::gLogInfo << "Detecting time: " <<
        //     general_config.duration.count() * 1000 << "ms" << std::endl;
        // }

        if (general_config.is_save &&
            tooth_detection_config.tooth_state_miss.count(
                tooth_detection_config
                    .max_class_label)) // 只在斗齿缺失时保存缺齿图像
        {
          save(yolo_->getParam().class_names, yolo_->getParam().save_path,
               image_processing_config, tooth_detection_config,
               pedestrian_detection_config, input_config, roi_config);
          sample::gLogInfo << "警告：检测到缺齿状态，已保存缺齿图像。"
                           << std::endl;
        }
        // 重置检测结果
        tooth_detection_config.match_cnt = 0;
        tooth_detection_config.max_count = 0;
        tooth_detection_config.max_class_label = "";
        tooth_detection_config.detect_res.clear();
      }
    }
    else // 铲斗状态信息漏检或者误检，那么在推理时不显示状态信息
    {
      tooth_detection_config.show_tooth_state_name = "";
    }

    // 未检测到缺齿超过1秒，解除大屏报警
    if (tooth_detection_config.is_tooth_detected)
    {
      auto current_time = std::chrono::steady_clock::now();
      // // 打印时间状态：
      // std::cout << "last_tooth_detected_time_ 是否有效: "
      //           << (tooth_detection_config.last_tooth_detected_time_ !=
      //           std::chrono::steady_clock::time_point())
      //           << std::endl;
      // std::cout << "时间差: "
      //           <<
      //           std::chrono::duration_cast<std::chrono::seconds>(current_time
      //           - tooth_detection_config.last_tooth_detected_time_).count()
      //           << "秒" << std::endl;
      if (tooth_detection_config.last_tooth_detected_time_ !=
              std::chrono::steady_clock::time_point() &&
          (current_time - tooth_detection_config.last_tooth_detected_time_) >
              std::chrono::seconds(1))
      {
        if (tooth_detection_config.teeth_root_coordinates.empty() &&
            mqtt_publisher_->isMqttConnected())
        {
          nlohmann::json payload_json;
          payload_json["toothDetection"] =
              tooth_detection_config.detection_mapping["complete"];
          std::string payload = payload_json.dump();

          // 使用异步发布（不再需要锁）
          if (!mqtt_publisher_->publishAsync(payload))
          {
            sample::gLogWarning << "MQTT Publish Failed!" << std::endl;
          }
          sample::gLogInfo << "MQTT Publish Success! Payload: " << payload
                           << std::endl;

          // 重置时间戳，避免重复发送0
          tooth_detection_config.last_tooth_detected_time_ =
              std::chrono::steady_clock::time_point();
          tooth_detection_config.is_tooth_detected = false;
        }
      }
    }

    // 每张图像检测完，重置容器
    tooth_detection_config.tooth_state_name.clear();
    tooth_detection_config.teeth_root_coordinates.clear();
    tooth_detection_config.root_idx.clear();
    pedestrian_detection_config.people_cnt = 0;
  }

  if (general_config.is_show)
  {
    show(yolo_->getParam().class_names, image_processing_config,
         tooth_detection_config, pedestrian_detection_config, input_config,
         roi_config, general_config);
  }

  // if (general_config.is_save)
  // {
  //     save(yolo_->getParam().class_names, yolo_->getParam().save_path,
  //     image_processing_config,
  //          tooth_detection_config, pedestrian_detection_config, input_config,
  //          roi_config);
  // }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  // if (input_config.idx == 0)
  //     sample::gLogInfo << "Source " << input_config.idx << " time used = " <<
  //     (1000 * time_used.count()) << " ms" << std::endl;
}

void yolo_detect::detect(std::unique_ptr<YOLOModel> &yolo_,
                         std::unique_ptr<YOLOModel> &second_yolo_,
                         std::unique_ptr<TensorRTInference> &trt_classifier_,
                         std::unique_ptr<UndistorterInterface> &undistorter_,
                         std::unique_ptr<MQTTPublish> &mqtt_publisher_,
                         std::shared_ptr<ROS2Client> &ros2_client_,
                         ModelConfig &model_config,
                         UndistortConfig &undistort_config,
                         InputConfig &input_config,
                         ToothDetectionConfig &tooth_detection_config,
                         PedestrianDetectionConfig &pedestrian_detection_config,
                         ImageProcessingConfig &image_processing_config,
                         GeneralConfig &general_config, ROIConfig &roi_config,
                         MqttConfig &mqtt_config)
{
  // 1. 先检查当前帧是否有效
  if (image_processing_config.init_frame.empty())
  {
    return; // 无效帧直接跳过
  }

  // 2. 统一处理当前帧（缩放+裁剪）
  if (image_processing_config.init_frame.size() !=
      input_config.input_sizes[input_config.idx])
  {
    cv::resize(image_processing_config.init_frame,
               image_processing_config.init_frame,
               input_config.input_sizes[input_config.idx]);
  }
  image_processing_config.cropped_frame = image_processing_config.init_frame(
      cv::Range(roi_config.roi_regions[input_config.idx][1],
                roi_config.roi_regions[input_config.idx][3]),
      cv::Range(roi_config.roi_regions[input_config.idx][0],
                roi_config.roi_regions[input_config.idx][2]));

  // 3. 无论batch是否满，都添加当前帧到batch
  image_processing_config.imgs_batch.emplace_back(
      image_processing_config.init_frame.clone());
  image_processing_config.crop_batch.emplace_back(
      image_processing_config.cropped_frame.clone());

  // 4. 检查batch是否已满，满则立即处理
  if (image_processing_config.imgs_batch.size() >=
      yolo_->getParam().batch_size)
  {
    std::lock_guard<std::mutex> lock(general_config.mtx);
    task(yolo_, second_yolo_, trt_classifier_, mqtt_publisher_, ros2_client_,
         model_config, tooth_detection_config, pedestrian_detection_config,
         image_processing_config, general_config, input_config, roi_config);
    image_processing_config.imgs_batch.clear();
    image_processing_config.crop_batch.clear();
    ++image_processing_config.batchi;
  }
}
