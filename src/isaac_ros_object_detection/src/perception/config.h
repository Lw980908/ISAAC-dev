#ifndef DETECTION_CONFIG_H
#define DETECTION_CONFIG_H

#include "thread_pool.h"

// OpenCV
#include <opencv2/opencv.hpp>

// C++标准库
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Box {
  float left, top, right, bottom, confidence;
  int label;
  std::vector<cv::Point2i> land_marks;

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence,
      int label)
      : left(left), top(top), right(right), bottom(bottom),
        confidence(confidence), label(label) {}

  Box(float left, float top, float right, float bottom, float confidence,
      int label, int numLandMarks)
      : left(left), top(top), right(right), bottom(bottom),
        confidence(confidence), label(label) {
    land_marks.reserve(numLandMarks);
  }
};

// ----------------------------- 斗齿检测数据 -----------------------------
enum class ExcavatorType {
  Front_shovel_excavator, // 正铲挖掘机
  Back_shovel_excavator   // 反铲挖掘机
};

struct ToothDetectionConfig {
  std::unordered_set<int> tooth_state = {4, 5, 6, 7, 8, 9, 10};
  std::unordered_set<int> teeth_and_root = {2, 3};
  std::unordered_multiset<std::string> tooth_state_name;
  std::unordered_set<std::string> tooth_state_miss = {
      "miss1", "miss2", "miss3", "miss4", "miss5", "miss6"};
  std::vector<std::tuple<int, int, int>> teeth_root_coordinates;
  std::vector<int> root_idx;
  std::unordered_map<std::string, int> detect_res;
  std::string max_class_label = "";
  int max_count = 0;
  int match_cnt = 0;
  std::string show_tooth_state_name = "";
  std::unordered_set<std::string> second_detection_labels = {
      "tooth", "root",  "complete", "miss1", "miss2",
      "miss3", "miss4", "miss5",    "miss6"};
  std::unordered_map<std::string, std::array<int, 6>> detection_mapping = {
      {"complete", {0, 0, 0, 0, 0, 0}}, {"miss1", {1, 0, 0, 0, 0, 0}},
      {"miss2", {0, 1, 0, 0, 0, 0}},    {"miss3", {0, 0, 1, 0, 0, 0}},
      {"miss4", {0, 0, 0, 1, 0, 0}},    {"miss5", {0, 0, 0, 0, 1, 0}},
      {"miss6", {0, 0, 0, 0, 0, 1}}};
  const int teeth_complete_number = 5; // 完整时斗齿的数量
  // ExcavatorType excavator_type = ExcavatorType::Front_shovel_excavator;
  ExcavatorType excavator_type = ExcavatorType::Back_shovel_excavator;
  bool need_bucket_projection_{false};      // 是否需要铲斗投影
  bool need_digging_start_location_{false}; // 是否需要挖掘起点

  // 斗齿检测报警自动解除
  std::chrono::steady_clock::time_point last_tooth_detected_time_;
  bool is_tooth_detected{false};
};

// ----------------------------- 行人检测数据 -----------------------------
struct PedestrianDetectionConfig {
  int people_cnt = 0;
  int show_people_cnt = 0;

  // 行人检测报警自动解除
  std::chrono::steady_clock::time_point last_people_detected_time_;
  bool is_people_detected{false};
};

// ----------------------------- 图像处理数据 -----------------------------
struct ImageProcessingConfig {
  std::vector<std::vector<Box>> bboxes;
  cv::Mat init_frame;
  cv::Mat cropped_frame;
  std::vector<cv::Mat> imgs_batch;
  std::vector<cv::Mat> crop_batch;
  int batchi = 0;
  int total_batches = 0;
  int delay_time = 1;
  std::vector<cv::VideoCapture> captures{
      cv::VideoCapture(), // 默认构造
      cv::VideoCapture(),
      cv::VideoCapture(),
      cv::VideoCapture(),
  };
  bool need_draw_detection_area_{true}; // 是否需要绘制检测区域
  bool need_show_detection_info_{true}; // 是否需要显示检测结果
  // FPS 计时器
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>
      lastTime_list = {std::chrono::high_resolution_clock::now(),
                       std::chrono::high_resolution_clock::now(),
                       std::chrono::high_resolution_clock::now(),
                       std::chrono::high_resolution_clock::now()};
  std::vector<int> frameCount_list = {0, 0, 0, 0};
  std::vector<double> fps_list = {0, 0, 0, 0};
};

// ----------------------------- 模型配置 -----------------------------

struct InitParameter {
  int num_class{80}; // coco
  std::vector<std::string> class_names;
  std::vector<std::string> input_output_names;

  bool dynamic_batch{true};
  size_t batch_size;
  int src_h, src_w;
  int dst_h, dst_w;
  float scale{255.f};
  float means[3] = {0.f, 0.f, 0.f};
  float stds[3] = {1.f, 1.f, 1.f};

  float iou_thresh;
  float conf_thresh;

  int topK{32}; // 前topK个结果
  std::string save_path;

  std::string winname = "TensorRT-Alpha";
  int char_width = 11;
  int det_info_render_width = 15;
  double font_scale = 0.6;
  bool is_show{false};
  bool is_save{false};
};

struct ModelConfig {
  std::vector<std::string> model_paths = {
      // "../../yolov11/model/xichang_tooth.trt",
      // "../../yolov11/model/suzhou_tooth_detection.trt",
      "/home/nvidia/models/best.trt", "/home/nvidia/models/best.trt",
      "/home/nvidia/models/best.trt", "/home/nvidia/models/best.trt",
      // "../../yolov11/model/people_around_blank.trt",
      // "../../yolov11/model/xichang_people_around.trt",
      // "../../yolov11/model/xichang_people_around.trt",
      // "../../yolov11/model/xichang_people_around.trt",
      // "../../yolov11/model/xichang_people_behind.trt",
  };
  std::string second_detection_model_path = "/home/nvidia/models/best.trt";
  std::string classify_model_path = "/home/nvidia/models/mobilenet_v2_dla_fp16.trt";

  bool need_classifier_{true};                         // 是否需要分类器
  std::shared_ptr<ThreadPool> classifier_thread_pool_; // 分类器线程池
  int classifier_thread_limit = 4; // 分类器线程池限制
  bool need_frame_skip_{false};    // 是否需要跳帧处理

  int set_params_times =
      0; // 设置参数次数，在第二次时为二次目标检测模型设置参数
  bool need_second_detection_{true}; // 是否需要二次目标检测
  std::shared_ptr<ThreadPool> bucket_thread_pool_; // 二次斗齿检测线程池
  int bucket_detection_thread_limit = 1; // 斗齿检测线程池限制
};

// ----------------------------- 去畸变配置 -----------------------------
enum class CameraType {
  Fish_eye_camera, // 鱼眼相机
  Pin_hole_camera  // 小孔相机
};
// Small-hole camera internal reference, fisheye camera internal reference
struct UndistortConfig { // 鱼眼相机参数组结构体
  struct Fisheye_internal_params {
    double fx, fy, cx, cy;
    double k1, k2, k3, k4;
  };
  std::vector<Fisheye_internal_params> Fisheye_param_groups = {
      // 第一组参数
      {510.1235520300, 510.3819927982, 959.3302431292, 766.8341659661,
       0.1352664590, -0.0349875000, -0.0012790949, 0.0008911534},

      // 第二组参数
      {510.1235520300, 510.3819927982, 959.3302431292, 766.8341659661,
       0.1352664590, -0.0349875000, -0.0012790949, 0.0008911534},

      // 第三组参数
      {510.1235520300, 510.3819927982, 959.3302431292, 766.8341659661,
       0.1352664590, -0.0349875000, -0.0012790949, 0.0008911534},

      // 第四组参数
      {510.1235520300, 510.3819927982, 959.3302431292, 766.8341659661,
       0.1352664590, -0.0349875000, -0.0012790949, 0.0008911534},
  };

  // 小孔相机参数组
  struct Pin_hole_internal_params {
    double fx, fy, cx, cy;
    double k1, k2, p1, p2, k3, k4, k5, k6;
  };
  std::vector<Pin_hole_internal_params> Pin_hole_param_groups = {
      // 第一组参数
      {1338.6205405053, 1338.9647746340, 1439.3752223563, 927.6968013624,
       0.8253965004, 0.1031405454, -0.0000492753, -0.0000974581, 0.0002600385,
       1.1875291416, 0.3089333315, 0.0088293173},

      // 第二组参数
      {1338.6205405053, 1338.9647746340, 1439.3752223563, 927.6968013624,
       0.8253965004, 0.1031405454, -0.0000492753, -0.0000974581, 0.0002600385,
       1.1875291416, 0.3089333315, 0.0088293173},

      // 第三组参数
      {1338.6205405053, 1338.9647746340, 1439.3752223563, 927.6968013624,
       0.8253965004, 0.1031405454, -0.0000492753, -0.0000974581, 0.0002600385,
       1.1875291416, 0.3089333315, 0.0088293173},

      // 第四组参数
      {1338.6205405053, 1338.9647746340, 1439.3752223563, 927.6968013624,
       0.8253965004, 0.1031405454, -0.0000492753, -0.0000974581, 0.0002600385,
       1.1875291416, 0.3089333315, 0.0088293173},
  };

  bool need_undistort_{false}; // 是否需要去畸变
  bool use_gpu_undistort_{true}; // 是否使用GPU去畸变，默认使用CPU去畸变
  std::vector<CameraType> camera_type_groups = {
      // CameraType::Pin_hole_camera,
      CameraType::Fish_eye_camera,
      CameraType::Fish_eye_camera,
      CameraType::Fish_eye_camera,
      CameraType::Fish_eye_camera,
  };
};

// ----------------------------- 视频与输入源配置 -----------------------------
enum class InputStream : uint8_t {
  IMAGE = 1 << 0,    // 0b0001
  VIDEO = 1 << 1,    // 0b0010
  CAMERA = 1 << 2,   // 0b0100
  VP_filter = 1 << 3 // 0b1000
};

enum class ProcessMethod {
  MULTI_THREADS_MULTI_SOURCES,       // 多线程多输入源
  SINGLE_THREAD_SINGLE_SOURCE,       // 单线程单输入源
  SHARED_MODEL_THREAD_MULTI_SOURCES, // 共享模型多线程多输入源
  CAMERA_COMBINED_WITH_LIDAR,        // 相机与雷达组合输入源
};

struct InputConfig {
  std::vector<std::string> video_paths = {
      "/home/nvidia/ISAAC/src/videos/1.mp4",
      "/home/nvidia/ISAAC/src/videos/2.mp4",
      "/home/nvidia/ISAAC/src/videos/3.mp4",
      "/home/nvidia/ISAAC/src/videos/4.mp4",
  };
  std::vector<std::string> image_paths = {
      "../../data/pictures/front.jpg",
      // "../../data/pictures/bottom.jpg",
      // "../../data/pictures/left.jpg",
      // "../../data/pictures/right.jpg",

  };
  std::vector<int> camera_ids = {4, 5, 6, 7};

  // NITROS 话题相关配置
  // 相机命名空间（对应 v4l2_camera_h264_multi.yaml 中的 namespace）
  std::vector<std::string> camera_namespaces = {"left", "right", "front",
                                                "back"};
  // NITROS 订阅话题前缀（例如：空字符串表示 /left/image_raw）
  std::string nitros_topic_prefix = "";
  // NITROS 发布话题前缀（例如：空字符串表示 /left/processed_image）
  std::string nitros_pub_topic_prefix = "";

  std::unordered_map<int, std::string> input_source = {
      {1 << 0, "IMAGE"},
      {1 << 1, "VIDEO"},
      {1 << 2, "CAMERA"},

      {1 << 3, "VP_filter"},
  };
  std::vector<cv::Size> input_sizes = {
      // cv::Size(2560, 1440),
      // cv::Size(2880, 1860),
      // cv::Size(640, 640),
      cv::Size(1920, 1536),
      cv::Size(1920, 1536),
      cv::Size(1920, 1536),
      cv::Size(1920, 1536),
  }; // 输入源单帧图像大小
  int idx = 0;                                // 视频或图片索引
  static const int max_input_source_nums = 4; // 最大输入源数量
  bool is_input_init{false};                  // 输入源参数初始化标志
  InputStream source{InputStream::VIDEO};     // 输入源类型
  ProcessMethod process_method{
      ProcessMethod::SHARED_MODEL_THREAD_MULTI_SOURCES}; // 处理方式
  bool ros2_enabled{false}; // 是否使用ros2订阅输入话题
  bool use_ros2_component_container{false}; // 是否使用ros2组件容器
};

// ----------------------------- ROI区域配置 -----------------------------
struct ROIConfig {
  std::vector<std::vector<int>> roi_regions = {
      // 斗齿
      // {0, 0, 2560, 1440},
      // 主视角
      // {0, 0, 640, 640},
      {0, 0, 1920, 1536},
      {0, 0, 1920, 1536},
      {0, 0, 1920, 1536},
      {0, 0, 1920, 1536},
      {700, 500, 1400, 1000},
      // 下视角
      {250, 550, 1600, 900},
      // 左视角
      {0, 500, 1670, 1536},
      // 右视角
      {250, 300, 1900, 1536},
      // 后视角
      // {0, 0, 640, 368},
  };
};

// ----------------------------- MQTT配置 -----------------------------
struct MqttConfig {
  bool mqtt_enabled{true}; // 是否启用MQTT发送检测结果
  const std::string MQTT_PUB_TOPIC = "perception/object_detection";
  const std::string MQTT_BROKER = "tcp://broker.emqx.io"; // 服务器地址
  const std::string MQTT_CLIENT_ID =
      "mqtt_client_"; // 客户端ID前段，后段接纳秒，生成唯一id
  const std::string MQTT_USERNAME = "vehicle_ecu";
  const std::string MQTT_PASSWORD = "111test";

  // 接收话题
  const std::string subscribe_topic =
      "touch_screen/alarm_deactivated";   // 订阅话题
  const std::string subscribe_field = ""; // 订阅字段名
};

// ----------------------------- 图像编/解码配置 -----------------------------
enum class EncodingFormat {
  IMAGE_RAW, // sensor_msgs::msg::Image 标准格式（BGR/RGB）
  NITROS_NV12 // NITROS NV12 协商话题格式（用于与 H264 编码器配合）
};

struct ImageEncoderDecoderConfig {
  EncodingFormat encoding_format_sub{
      EncodingFormat::IMAGE_RAW}; // 订阅话题编码格式
  EncodingFormat encoding_format_pub{
      EncodingFormat::IMAGE_RAW}; // 发布话题编码格式
};

inline bool use_nitros_sub_ = false;

// ----------------------------- 其他通用配置（全局变量）
// -----------------------------
struct GeneralConfig {
  std::chrono::time_point<std::chrono::high_resolution_clock> start =
      std::chrono::high_resolution_clock::now();
  std::chrono::time_point<std::chrono::high_resolution_clock> end =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = std::chrono::duration<double>(0);
  std::vector<std::thread> threads; // 多线程处理
  std::mutex mtx;
  bool is_show{false};
  bool is_save{false};
  bool is_init{false}; // 参数初始化标志
  cv::Mat combined =
      cv::Mat::zeros(720, 1280, CV_8UC3); // 多线程合并显示图像（全黑背景）
  std::atomic<bool> exit_flag{false}; // 添加原子退出标志
  bool show_gui{true};                // 是否显示GUI窗口
};

// ----------------------------- 函数声明 -----------------------------
bool openInputStream(ImageProcessingConfig &image_processing_config,
                     InputConfig &input_config); // 打开输入流
void setRenderWindow(InitParameter &param);      // 设置渲染窗口
void setInputStream(ImageProcessingConfig &image_processing_config,
                    InputConfig &input_config, InitParameter &param);

void show(const std::vector<std::string> &classNames,
          ImageProcessingConfig &image_processing_config,
          ToothDetectionConfig &tooth_detection_config,
          PedestrianDetectionConfig &pedestrian_detection_config,
          InputConfig &input_config, ROIConfig &roi_config,
          GeneralConfig &general_config);

cv::Mat
draw_detection_boxes(const std::vector<std::string> &classNames,
                     ImageProcessingConfig &image_processing_config,
                     ToothDetectionConfig &tooth_detection_config,
                     PedestrianDetectionConfig &pedestrian_detection_config,
                     InputConfig &input_config, ROIConfig &roi_config);

void save(const std::vector<std::string> &classNames,
          const std::string &savePath,
          ImageProcessingConfig &image_processing_config,
          ToothDetectionConfig &tooth_detection_config,
          PedestrianDetectionConfig &pedestrian_detection_config,
          InputConfig &input_config, ROIConfig &roi_config);

#endif // DETECTION_CONFIG_H
