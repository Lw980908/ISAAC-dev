#include "yolo_utils.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>

template <typename T> T clamp(T value, T min_val, T max_val) {
  return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}
// 获取当前时间戳，并格式化为字符串
std::string getCurrentTimestamp() {
  auto now = std::time(nullptr);
  return std::to_string(now);
}

using namespace std::chrono;

// 获取当前日期，并格式化为 YYYY-MM-DD 形式的字符串
std::string getCurrentDate() {
  auto now = system_clock::now();
  auto now_time = system_clock::to_time_t(now);
  std::tm tm_date = *std::localtime(&now_time);

  std::ostringstream oss;
  oss << std::put_time(&tm_date, "%Y-%m-%d");
  return oss.str();
}
void yolo_utils::saveBinaryFile(float *vec, size_t len,
                                const std::string &file) {
  std::ofstream out(file, std::ios::out | std::ios::binary);
  if (!out.is_open())
    return;
  out.write((const char *)vec, sizeof(float) * len);
  out.close();
}

std::vector<uint8_t> yolo_utils::readBinaryFile(const std::string &file) {

  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open())
    return {};

  in.seekg(0, std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

std::vector<unsigned char> yolo_utils::loadModel(const std::string &file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    return {};
  }
  in.seekg(0, std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);
    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

std::string yolo_utils::getSystemTimeStr() {
  return std::to_string(std::rand());
}

void setInputStream(ImageProcessingConfig &image_processing_config,
                    InputConfig &input_config, InitParameter &param) {
  int total_frames = 0;
  std::string img_format;
  cv::VideoCapture capture;

  switch (input_config.source) {
  case InputStream::VP_filter:
    // image_processing_config.captures[input_config.idx].open(input_config.video_paths[input_config.idx]);
    // if (!image_processing_config.captures[input_config.idx].isOpened())
    // {
    // 	sample::gLogError << "Failed to open video file: " <<
    // input_config.video_paths[input_config.idx] << std::endl; 	return;
    // }
    param.batch_size = 1;
    total_frames = 1;
    image_processing_config.total_batches = 1;
    break;

  case InputStream::IMAGE:
    // img_format = imagePath.substr(imagePath.size() - 4, 4);
    img_format = input_config.image_paths[input_config.idx].substr(
        input_config.image_paths[input_config.idx].size() - 4, 4);
    if (img_format == ".png" || img_format == ".PNG") {
      sample::gLogWarning
          << "+-----------------------------------------------------------+"
          << std::endl;
      sample::gLogWarning
          << "| If you use PNG format pictures, the file name must be eg: |"
          << std::endl;
      sample::gLogWarning
          << "| demo0.png, demo1.png, demo2.png ......, but not demo.png. |"
          << std::endl;
      sample::gLogWarning << "| The above rules are determined by "
                             "OpenCV.					|"
                          << std::endl;
      sample::gLogWarning
          << "+-----------------------------------------------------------+"
          << std::endl;
    }
    // image_processing_config.captures[input_config.idx].open(input_config.image_paths[input_config.idx]);
    // // Open image file as a video stream if
    // (!image_processing_config.captures[input_config.idx].isOpened())
    // {
    // 	sample::gLogError << "Failed to open image file: " <<
    // input_config.image_paths[input_config.idx] << std::endl; 	return;
    // }
    param.batch_size = 1;
    total_frames = 1;
    image_processing_config.total_batches = 1;
    image_processing_config.delay_time = 0;
    break;

  case InputStream::VIDEO:
    // image_processing_config.captures[input_config.idx].open(input_config.video_paths[input_config.idx]);
    // if (!image_processing_config.captures[input_config.idx].isOpened())
    // {
    // 	sample::gLogError << "Failed to open video file: " <<
    // input_config.video_paths[input_config.idx] << std::endl; 	return;
    // }
    capture.open(input_config.video_paths[input_config.idx]);
    total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    image_processing_config.total_batches =
        (total_frames % param.batch_size == 0)
            ? (total_frames / param.batch_size)
            : (total_frames / param.batch_size + 1);
    capture.release();
    if (total_frames <= 0)
      break;

  case InputStream::CAMERA:
    // image_processing_config.captures[input_config.idx].open(input_config.camera_ids[input_config.idx],
    // cv::CAP_V4L2); // 显式指定 V4L2 后端 if
    // (!image_processing_config.captures[input_config.idx].isOpened())
    // {
    // 	sample::gLogError << "Failed to open camera with V4L2 backend!" <<
    // std::endl; 	return;
    // }
    // // 设置摄像头参数
    // image_processing_config.captures[input_config.idx].set(cv::CAP_PROP_FRAME_WIDTH,
    // input_config.input_sizes[input_config.idx].width);
    // image_processing_config.captures[input_config.idx].set(cv::CAP_PROP_FRAME_HEIGHT,
    // input_config.input_sizes[input_config.idx].height);
    // image_processing_config.captures[input_config.idx].set(cv::CAP_PROP_FPS,
    // 30);
    total_frames = INT_MAX;
    image_processing_config.total_batches = INT_MAX;
    break;

  default:
    break;
  }
}

bool openInputStream(ImageProcessingConfig &image_processing_config,
                     InputConfig &input_config) {
  switch (input_config.source) {
  case InputStream::VP_filter:
    image_processing_config.captures[input_config.idx].open(
        input_config.video_paths[input_config.idx]);
    if (!image_processing_config.captures[input_config.idx].isOpened()) {
      sample::gLogError << "Failed to open video file: "
                        << input_config.video_paths[input_config.idx]
                        << std::endl;
      return false;
    }
    break;

  case InputStream::IMAGE:
    image_processing_config.captures[input_config.idx].open(
        input_config.image_paths[input_config.idx]); // Open image file as a
                                                     // video stream
    if (!image_processing_config.captures[input_config.idx].isOpened()) {
      sample::gLogError << "Failed to open image file: "
                        << input_config.image_paths[input_config.idx]
                        << std::endl;
      return false;
    }
    break;

  case InputStream::VIDEO:
    image_processing_config.captures[input_config.idx].open(
        input_config.video_paths[input_config.idx]);
    if (!image_processing_config.captures[input_config.idx].isOpened()) {
      sample::gLogError << "Failed to open video file: "
                        << input_config.video_paths[input_config.idx]
                        << std::endl;
      return false;
    }
    break;

  case InputStream::CAMERA: {
    // 使用 V4L2 后端打开摄像头
    image_processing_config.captures[input_config.idx].open(
        input_config.camera_ids[input_config.idx], cv::CAP_V4L2);
    if (!image_processing_config.captures[input_config.idx].isOpened()) {
      sample::gLogError << "Failed to open camera!" << std::endl;
      return false;
    }
    // 设置参数（V4L2 回退时）
    image_processing_config.captures[input_config.idx].set(
        cv::CAP_PROP_FRAME_WIDTH,
        input_config.input_sizes[input_config.idx].width);
    image_processing_config.captures[input_config.idx].set(
        cv::CAP_PROP_FRAME_HEIGHT,
        input_config.input_sizes[input_config.idx].height);
    image_processing_config.captures[input_config.idx].set(cv::CAP_PROP_FPS,
                                                           30);
    sample::gLogInfo << "Camera opened successfully with V4L2 backend!"
                     << std::endl;

    break;
  }

  default:
    sample::gLogError << "Unsupported input stream type!" << std::endl;
    return false;
  }
  return true;
}

void setRenderWindow(InitParameter &param) {
  if (!param.is_show)
    return;
  int max_w = 960;
  int max_h = 540;
  float scale_h = (float)param.src_h / max_h;
  float scale_w = (float)param.src_w / max_w;
  if (scale_h > 1.f && scale_w > 1.f) {
    float scale = scale_h < scale_w ? scale_h : scale_w;
    cv::namedWindow(param.winname,
                    cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO); // for Linux
    cv::resizeWindow(param.winname, int(param.src_w / scale),
                     int(param.src_h / scale));
    param.char_width = 16;
    param.det_info_render_width = 18;
    param.font_scale = 0.9;
  } else {
    cv::namedWindow(param.winname);
  }
}

std::string yolo_utils::getTimeStamp() {
  std::chrono::nanoseconds t =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch());
  return std::to_string(t.count());
}

void drawDashedRectangle(cv::Mat &img, const std::vector<int> &roi_region) {
  // 定义虚线的间隔长度
  int dash_length = 10; // 虚线段的长度
  int gap_length = 10;  // 虚线之间的间隔

  // 获取矩形的坐标
  int x1 = roi_region[0];
  int y1 = roi_region[1];
  int x2 = roi_region[2];
  int y2 = roi_region[3];

  // 上边框
  for (int i = x1; i < x2; i += (dash_length + gap_length)) {
    cv::Point start(i, y1);
    cv::Point end(std::min(i + dash_length, x2), y1);
    cv::line(img, start, end, cv::Scalar(0, 255, 255), 2, cv::LINE_AA); // 黄色
  }

  // 下边框
  for (int i = x1; i < x2; i += (dash_length + gap_length)) {
    cv::Point start(i, y2 - 1);
    cv::Point end(std::min(i + dash_length, x2), y2 - 1);
    cv::line(img, start, end, cv::Scalar(0, 255, 255), 2, cv::LINE_AA); // 黄色
  }

  // 左边框
  for (int i = y1; i < y2; i += (dash_length + gap_length)) {
    cv::Point start(x1, i);
    cv::Point end(x1, std::min(i + dash_length, y2));
    cv::line(img, start, end, cv::Scalar(0, 255, 255), 2, cv::LINE_AA); // 黄色
  }

  // 右边框
  for (int i = y1; i < y2; i += (dash_length + gap_length)) {
    cv::Point start(x2 - 1, i);
    cv::Point end(x2 - 1, std::min(i + dash_length, y2));
    cv::line(img, start, end, cv::Scalar(0, 255, 255), 2, cv::LINE_AA); // 黄色
  }
}

std::string savePath = "../yolov11/save";
void show(const std::vector<std::string> &classNames,
          ImageProcessingConfig &image_processing_config,
          ToothDetectionConfig &tooth_detection_config,
          PedestrianDetectionConfig &pedestrian_detection_config,
          InputConfig &input_config, ROIConfig &roi_config,
          GeneralConfig &general_config) {
  // 计算最佳网格布局（适配1-4个源）
  int num_sources = std::max({input_config.image_paths.size(),
                              input_config.video_paths.size(),
                              input_config.camera_ids.size()});
  int grid_rows = 1, grid_cols = 1;
  if (num_sources == 2) {
    grid_rows = 1;
    grid_cols = 2;
  } else if (num_sources >= 3) {
    grid_rows = 2;
    grid_cols = 2;
  }

  // 2. 计算每个子图尺寸
  int sub_width = general_config.combined.cols / grid_cols;
  int sub_height = general_config.combined.rows / grid_rows;

  cv::Scalar color = cv::Scalar(0, 255, 0);
  cv::Point bbox_points[1][4];
  const cv::Point *bbox_point0[1] = {bbox_points[0]};
  int num_points[] = {4};

  for (size_t bi = 0; bi < image_processing_config.imgs_batch.size(); bi++) {
    // 计算 FPS
    ++image_processing_config.frameCount_list[input_config.idx];
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed =
        currentTime - image_processing_config.lastTime_list[input_config.idx];
    // 在右上角显示 FPS
    cv::putText(
        image_processing_config.imgs_batch[bi],
        "FPS: " +
            std::to_string(image_processing_config.fps_list[input_config.idx]),
        cv::Point(image_processing_config.imgs_batch[bi].cols - 310, 50),
        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    if (elapsed.count() >= 1.0) // 每一秒更新一次
    {
      image_processing_config.fps_list[input_config.idx] =
          image_processing_config.frameCount_list[input_config.idx] /
          elapsed.count();
      image_processing_config.frameCount_list[input_config.idx] = 0;
      image_processing_config.lastTime_list[input_config.idx] = currentTime;
    }
    // 显示人数
    if (pedestrian_detection_config.show_people_cnt) {
      cv::putText(
          image_processing_config.imgs_batch[bi],
          "people_count: " +
              std::to_string(pedestrian_detection_config.show_people_cnt),
          cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255),
          2, cv::LINE_AA);
    }

    if (!tooth_detection_config.show_tooth_state_name.empty()) {
      cv::putText(
          image_processing_config.imgs_batch[bi],
          "root_detect: " + tooth_detection_config.show_tooth_state_name,
          cv::Point(image_processing_config.imgs_batch[bi].cols / 2 - 200, 50),
          cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // 确定每张图像的ROI偏移
    int roiOffsetX = roi_config.roi_regions[input_config.idx][0];
    int roiOffsetY = roi_config.roi_regions[input_config.idx][1];
    // 绘制检测区域
    drawDashedRectangle(image_processing_config.imgs_batch[bi],
                        roi_config.roi_regions[input_config.idx]);
    if (!image_processing_config.bboxes.empty()) {
      for (auto &box : image_processing_config.bboxes[bi]) {
        if (classNames.size() == 15) {
          color = yolo_utils::Colors::colors_universal[box.label];
        }

        if (tooth_detection_config.tooth_state.count(box.label) ||
            box.label == 0) {
          if (box.label == 0) {
            cv::rectangle(
                image_processing_config.imgs_batch[bi],
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY),
                cv::Point(box.right + roiOffsetX, box.bottom + roiOffsetY),
                color, 2, cv::LINE_AA);
          }

          else if (image_processing_config.need_show_detection_info_ &&
                   !tooth_detection_config.show_tooth_state_name.empty()) {
            cv::rectangle(
                image_processing_config.imgs_batch[bi],
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY),
                cv::Point(box.right + roiOffsetX, box.bottom + roiOffsetY),
                color, 2, cv::LINE_AA);
            // 字体框
            cv::String det_info = tooth_detection_config.show_tooth_state_name +
                                  " " + cv::format("%.2f", box.confidence);
            bbox_points[0][0] =
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY);
            bbox_points[0][1] =
                cv::Point(box.left + det_info.size() * 18 + roiOffsetX,
                          box.top + roiOffsetY);
            bbox_points[0][2] =
                cv::Point(box.left + det_info.size() * 18 + roiOffsetX,
                          box.top + roiOffsetY - 23);
            bbox_points[0][3] =
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY - 23);
            cv::fillPoly(image_processing_config.imgs_batch[bi], bbox_point0,
                         num_points, 1, color);
            cv::putText(image_processing_config.imgs_batch[bi], det_info,
                        bbox_points[0][0], cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
          }
        }
        if (tooth_detection_config.teeth_and_root.count(
                box.label)) // 斗齿和齿根单独处理
        {
          cv::Mat create_alpha = cv::Mat::zeros(
              image_processing_config.imgs_batch[bi].rows,
              image_processing_config.imgs_batch[bi].cols, CV_8UC3);
          create_alpha.setTo(color);
          cv::Point pt1(box.left + roiOffsetX, box.top + roiOffsetY);
          cv::Point pt2(box.right + roiOffsetX, box.bottom + roiOffsetY);
          pt1.x = std::max(0, pt1.x);
          pt1.y = std::max(0, pt1.y);
          pt2.x = std::min(image_processing_config.imgs_batch[bi].cols, pt2.x);
          pt2.y = std::min(image_processing_config.imgs_batch[bi].rows, pt2.y);
          cv::Mat img_add;
          cv::addWeighted(
              image_processing_config.imgs_batch[bi](cv::Rect(pt1, pt2)), 0.7,
              create_alpha(cv::Rect(pt1, pt2)), 0.3, 0, img_add);
          img_add.copyTo(
              image_processing_config.imgs_batch[bi](cv::Rect(pt1, pt2)));
        }

        if (!box.land_marks.empty()) {
          for (auto &pt : box.land_marks) {
            cv::circle(image_processing_config.imgs_batch[bi],
                       pt + cv::Point(roiOffsetX, roiOffsetY), 1,
                       cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
          }
        }
      }
    }

    // 缩放到子图尺寸并放置到网格位置
    cv::Mat &frame = image_processing_config.imgs_batch[bi];
    if (frame.empty()) {
      frame = cv::Mat::zeros(sub_height, sub_width, CV_8UC3);
    } else {
      cv::resize(frame, frame, cv::Size(sub_width, sub_height));
    }

    // 计算网格位置
    int row = input_config.idx / grid_cols;
    int col = input_config.idx % grid_cols;
    cv::Rect roi(col * sub_width, row * sub_height, sub_width, sub_height);

    // 将图像复制到对应位置
    frame.copyTo(general_config.combined(roi));

    // 添加源ID标注
    cv::putText(general_config.combined,
                "Source_" + std::to_string(input_config.idx),
                cv::Point(col * sub_width + 10, row * sub_height + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
  }

  // std::string timestamp = getCurrentTimestamp();
  // std::string filename = savePath + (savePath.back() == '/' ? "" : "/") +
  // timestamp + ".jpg"; cv::imwrite(filename, general_config.combined);

  cv::imshow("Multi-Source Display", general_config.combined);
  char key = static_cast<char>(cv::waitKey(image_processing_config.delay_time));
  if (key == 27) {
    general_config.exit_flag.store(true); // 设置全局退出标志
  }
}

cv::Mat
draw_detection_boxes(const std::vector<std::string> &classNames,
                     ImageProcessingConfig &image_processing_config,
                     ToothDetectionConfig &tooth_detection_config,
                     PedestrianDetectionConfig &pedestrian_detection_config,
                     InputConfig &input_config, ROIConfig &roi_config) {
  cv::Scalar color = cv::Scalar(0, 255, 0);
  cv::Point bbox_points[1][4];
  const cv::Point *bbox_point0[1] = {bbox_points[0]};
  int num_points[] = {4};
  auto &img = image_processing_config.init_frame;
  auto &bboxes = image_processing_config.bboxes;

  // 绘制ROI区域
  int roiOffsetX = roi_config.roi_regions[input_config.idx][0];
  int roiOffsetY = roi_config.roi_regions[input_config.idx][1];
  if (image_processing_config.need_draw_detection_area_) {
    drawDashedRectangle(img, roi_config.roi_regions[input_config.idx]);
  }
  if (image_processing_config.need_show_detection_info_) { // 计算 FPS
    ++image_processing_config.frameCount_list[input_config.idx];
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed =
        currentTime - image_processing_config.lastTime_list[input_config.idx];
    // 在右上角显示 FPS
    cv::putText(
        img,
        "FPS: " +
            std::to_string(image_processing_config.fps_list[input_config.idx]),
        cv::Point(img.cols - 310, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    if (elapsed.count() >= 1.0) // 每一秒更新一次
    {
      image_processing_config.fps_list[input_config.idx] =
          image_processing_config.frameCount_list[input_config.idx] /
          elapsed.count();
      image_processing_config.frameCount_list[input_config.idx] = 0;
      image_processing_config.lastTime_list[input_config.idx] = currentTime;
    }
    // 显示人数
    if (pedestrian_detection_config.show_people_cnt) {
      cv::putText(
          img,
          "people_count: " +
              std::to_string(pedestrian_detection_config.show_people_cnt),
          cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255),
          2, cv::LINE_AA);
    }

    // 显示斗齿检测结果
    if (!tooth_detection_config.show_tooth_state_name.empty()) {
      cv::putText(
          img, "root_detect: " + tooth_detection_config.show_tooth_state_name,
          cv::Point(img.cols / 2 - 200, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
          cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
  }

  // 如果有检测结果
  if (!bboxes.empty()) {
    for (auto &box : bboxes[0]) {
      if (classNames.size() == 15) {
        color = yolo_utils::Colors::colors_universal[box.label];
      }

      // 行人/铲斗检测框绘制
      if (tooth_detection_config.tooth_state.count(box.label) ||
          box.label == 0) {
        // 计算原始坐标
        int left = box.left + roiOffsetX;
        int top = box.top + roiOffsetY;
        int right = box.right + roiOffsetX;
        int bottom = box.bottom + roiOffsetY;

        // 边界检查与调整
        left = std::max(0, left);
        top = std::max(0, top);
        right = std::min(img.cols, right);
        bottom = std::min(img.rows, bottom);

        // 检查有效矩形
        if (right <= left || bottom <= top) {
          continue;
        }

        if (box.label == 0) {
          // 行人检测框
          cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom),
                        color, 2, cv::LINE_AA);
        } else if (image_processing_config.need_show_detection_info_ &&
                   !tooth_detection_config.show_tooth_state_name.empty()) {
          // 铲斗检测框
          cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom),
                        color, 2, cv::LINE_AA);
          cv::String det_info = tooth_detection_config.show_tooth_state_name +
                                " " + cv::format("%.2f", box.confidence);
          // 计算文本框位置
          int text_x = left;
          int text_y = top - 23;
          int text_w = det_info.size() * 18;
          int text_h = 23;

          // 文本框边界检查
          text_x = std::max(0, text_x);
          text_y = std::max(0, text_y);
          text_w = std::min(text_w, img.cols - text_x);
          text_h = std::min(text_h, img.rows - text_y);

          if (text_w > 0 && text_h > 0) {
            // 绘制文本背景
            cv::rectangle(img, cv::Point(text_x, text_y),
                          cv::Point(text_x + text_w, text_y + text_h), color,
                          cv::FILLED);
            // 绘制文本
            cv::putText(img, det_info, cv::Point(text_x, text_y + text_h - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255),
                        2, cv::LINE_AA);
          }
        }
      }

      // 斗齿和齿根处理
      if (tooth_detection_config.teeth_and_root.count(box.label)) {
        int left = box.left + roiOffsetX;
        int top = box.top + roiOffsetY;
        int right = box.right + roiOffsetX;
        int bottom = box.bottom + roiOffsetY;

        // 边界检查与调整
        left = std::max(0, left);
        top = std::max(0, top);
        right = std::min(img.cols, right);
        bottom = std::min(img.rows, bottom);

        // 检查有效矩形
        if (right <= left || bottom <= top) {
          continue;
        }

        cv::Mat create_alpha = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
        create_alpha.setTo(color);

        // 创建安全ROI
        cv::Rect roi_rect(left, top, right - left, bottom - top);
        cv::Mat img_roi = img(roi_rect);
        cv::Mat alpha_roi = create_alpha(roi_rect);

        // 确保ROI有效
        if (!img_roi.empty() && !alpha_roi.empty() &&
            img_roi.rows == alpha_roi.rows && img_roi.cols == alpha_roi.cols) {
          cv::Mat img_add;
          cv::addWeighted(img_roi, 0.7, alpha_roi, 0.3, 0, img_add);
          img_add.copyTo(img_roi);
        }
      }

      if (tooth_detection_config.need_bucket_projection_ &&
          tooth_detection_config.excavator_type ==
              ExcavatorType::Front_shovel_excavator &&
          box.label == 1) {
        // -------------------------绘制铲斗投影---------------------------------
        // 定义倾斜平面（示例：左下角深50像素）
        // std::vector<int> projection_region = {1192, 755, 1290, 1100};
        std::vector<int> projection_region = {0, 0, 100, 1100};
        cv::Point3f plane_tl(projection_region[0], projection_region[1], 0);
        cv::Point3f plane_tr(projection_region[2], projection_region[1], 0);
        cv::Point3f plane_bl(projection_region[0], projection_region[3], 50);

        // 计算平面方程 Ax + By + Cz + D = 0
        cv::Point3f vec1 = plane_tr - plane_tl;
        cv::Point3f vec2 = plane_bl - plane_tl;
        cv::Point3f normal = vec1.cross(vec2);
        normal /= cv::norm(normal);
        float A = normal.x, B = normal.y, C = normal.z;
        float D = -(A * plane_tl.x + B * plane_tl.y + C * plane_tl.z);

        // 铲斗3D位置（假设高度H=100像素）
        cv::Point2f box_center_2d((box.left + box.right) / 2 +
                                      roiOffsetX, // X坐标：检测框中心
                                  box.bottom + roiOffsetY); // Y坐标：检测框底部
        float H = 100.0f;
        cv::Point3f box_3d(box_center_2d.x, box_center_2d.y, H);

        // 计算视线方向
        cv::Point3f dir = box_3d - cv::Point3f(0, 0, 0);
        dir /= cv::norm(dir);

        // 求视线与平面交点
        float t =
            -(A * 0 + B * 0 + C * 0 + D) / (A * dir.x + B * dir.y + C * dir.z);
        cv::Point3f intersect_3d = dir * t;
        cv::Point2f proj_point(intersect_3d.x, intersect_3d.y);

        // 计算对角线的起点和终点
        cv::Point2f diagonal_start(projection_region[0], projection_region[1]);
        cv::Point2f diagonal_end(projection_region[2], projection_region[3]);

        // 计算对角线的向量
        cv::Point2f diagonal_vec = diagonal_end - diagonal_start;

        // 根据宽度调整物体的深度（距离摄像头的距离）
        float box_width = box.right - box.left;
        float width_factor =
            1.0f / (1.0f + box_width * 0.015f); // 增大系数，增强宽度影响
        // 根据检测框的纵坐标调整物体与地面的垂直距离（物体越高离地越远）
        float box_height = box.bottom - box.top;
        float height_factor = std::pow(1.0f / (1.0f + box_height * 0.01f),
                                       0.5f); // 调整幂次和系数
        // 计算深度因子，假设通过 Z 轴深度进行计算
        float depth_factor =
            1.0f / (1.0f + intersect_3d.z * 0.002f); // 减小系数，提升深度影响
        // 调整权重分配（深度因子占更大比例）
        float scale_factor =
            0.3f * width_factor + 0.2f * height_factor + 0.5f * depth_factor;
        scale_factor =
            clamp(scale_factor, 0.1f, 1.0f); // 确保比例因子在合理范围内
        // std::cout << "scale_factor: " << scale_factor << std::endl;
        // std::cout << "---------------------------------- " << std::endl;

        // 将投影点映射到对角线上的位置
        float diagonal_length = cv::norm(diagonal_vec);
        float proj_length = diagonal_length * scale_factor;
        cv::Point2f proj_on_diagonal =
            diagonal_start + diagonal_vec * (proj_length / diagonal_length);

        // 计算投影点的对应横向水平线
        float ellipse_width = box.right - box.left;
        int ellipse_height = ellipse_width * scale_factor; // 调整纵向尺寸比例
        float angle =
            std::atan2(plane_tr.y - plane_tl.y, plane_tr.x - plane_tl.x) * 180 /
            CV_PI;

        // 计算椭圆中心
        cv::Point2f ellipse_center(box_center_2d.x, proj_on_diagonal.y);

        // 确保椭圆始终在检测框下方
        if (ellipse_center.y < box.bottom + roiOffsetY) {
          ellipse_center.y =
              box.bottom + roiOffsetY; // 强制椭圆投影在框的下方
                                       // 使椭圆更加扁平，通过减小纵向尺寸
          float flattened_height = ellipse_height * 0.5f; // 将高度减小50%

          // 确保椭圆的上边缘不碰到检测框
          if (ellipse_center.y - flattened_height / 2 <
              box.bottom + roiOffsetY) {
            // 调整椭圆的中心位置，确保它不碰到框的顶部
            ellipse_center.y = box.bottom + roiOffsetY + flattened_height / 2;
          }

          // 绘制椭圆（扁平化后的椭圆）
          cv::Mat overlay;
          img.copyTo(overlay);
          cv::ellipse(overlay, ellipse_center,
                      cv::Size(ellipse_width / 2, flattened_height / 2), angle,
                      0, 360, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
          cv::addWeighted(overlay, 0.2, img, 0.8, 0, img);
          // // 绘制椭圆边框
          // cv::ellipse(img, ellipse_center,
          // 			cv::Size(ellipse_width / 2, flattened_height /
          // 2), 			angle, 0, 360,
          // cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        } else { // 绘制椭圆
          cv::Mat overlay;
          img.copyTo(overlay);
          cv::ellipse(overlay, ellipse_center,
                      cv::Size(ellipse_width / 2, ellipse_height / 2), angle, 0,
                      360, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
          cv::addWeighted(overlay, 0.2, img, 0.8, 0, img);
          // // 绘制椭圆轮廓
          // cv::ellipse(img, ellipse_center,
          // 			cv::Size(ellipse_width / 2, ellipse_height / 2),
          // 			angle, 0, 360,
          // 			cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
        // //
        // -----------------------------------绘制水平红色虚线-----------------------------------
        // // 随铲斗动
        // cv::Point line_start(
        // 	box.right + roiOffsetX,
        // 	box.top + roiOffsetY + 50 // 顶部Y
        // );
        // int line_length = 100;			  // 虚线总长度
        // int dash_length = 10;			  // 单段虚线长度
        // int gap_length = 5;				  // 间隔长度
        // cv::Scalar line_color(0, 0, 255); // BGR红色

        // // 分段绘制
        // int drawn_length = 0;
        // while (drawn_length < line_length)
        // {
        // 	// 计算当前段终点
        // 	cv::Point line_end = line_start + cv::Point(
        // 										  std::min(dash_length,
        // line_length - drawn_length),
        // 0);

        // 	// 绘制实线部分
        // 	if (line_start.x < img.cols && line_end.x < img.cols)
        // 	{
        // 		cv::line(img, line_start, line_end, line_color, 2,
        // cv::LINE_AA);
        // 	}

        // 	// 更新位置
        // 	drawn_length += (dash_length + gap_length);
        // 	line_start.x += (dash_length + gap_length);
        // }
      } else if (tooth_detection_config.need_bucket_projection_ &&
                 tooth_detection_config.excavator_type ==
                     ExcavatorType::Back_shovel_excavator &&
                 box.label == 1) {
        // -------------------------绘制铲斗投影---------------------------------
        // 定义倾斜平面
        std::vector<int> projection_region = {1192, 755, 1290, 1100};
        cv::Point3f plane_tl(projection_region[0], projection_region[1], 0);
        cv::Point3f plane_tr(projection_region[2], projection_region[1], 0);
        cv::Point3f plane_bl(projection_region[0], projection_region[3], 50);

        // 计算平面方程
        cv::Point3f vec1 = plane_tr - plane_tl;
        cv::Point3f vec2 = plane_bl - plane_tl;
        cv::Point3f normal = vec1.cross(vec2);
        normal /= cv::norm(normal);
        float A = normal.x, B = normal.y, C = normal.z;
        float D = -(A * plane_tl.x + B * plane_tl.y + C * plane_tl.z);

        // 铲斗3D位置（假设高度H=100像素）
        cv::Point2f box_center_2d((box.left + box.right) / 2 + roiOffsetX,
                                  box.top + roiOffsetY);
        float H = 100.0f;
        cv::Point3f box_3d(box_center_2d.x, box_center_2d.y, H);

        // 计算视线方向
        cv::Point3f dir = box_3d - cv::Point3f(0, 0, 0);
        dir /= cv::norm(dir);

        // 求视线与平面交点
        float t =
            -(A * 0 + B * 0 + C * 0 + D) / (A * dir.x + B * dir.y + C * dir.z);
        cv::Point3f intersect_3d = dir * t;
        cv::Point2f proj_point(intersect_3d.x, intersect_3d.y);

        // 计算对角线向量
        cv::Point2f diagonal_start(projection_region[0], projection_region[1]);
        cv::Point2f diagonal_end(projection_region[2], projection_region[3]);
        cv::Point2f diagonal_vec = diagonal_end - diagonal_start;

        // 根据宽度调整物体的深度（距离摄像头的距离）
        float box_width = box.right - box.left;
        float width_factor =
            1.0f / (1.0f + box_width * 0.015f); // 增大系数，增强宽度影响
        // 根据检测框的纵坐标调整物体与地面的垂直距离（物体越高离地越远）
        float box_height = box.bottom - box.top;
        float height_factor = std::pow(1.0f / (1.0f + box_height * 0.01f),
                                       0.5f); // 调整幂次和系数
        // 计算深度因子，假设通过 Z 轴深度进行计算
        float depth_factor =
            1.0f / (1.0f + intersect_3d.z * 0.002f); // 减小系数，提升深度影响
        // 调整权重分配（深度因子占更大比例）
        float scale_factor =
            0.3f * width_factor + 0.2f * height_factor + 0.5f * depth_factor;
        scale_factor =
            clamp(scale_factor, 0.1f, 1.0f); // 确保比例因子在合理范围内

        // 修改后的 scale_factor 权重
        // float scale_factor = 0.2f * width_factor + 0.3f * height_factor +
        // 0.5f * depth_factor;
        std::cout << "width_factor: " << width_factor << std::endl;
        std::cout << "height_factor: " << height_factor << std::endl;
        std::cout << "depth_factor: " << depth_factor << std::endl;
        // float scale_factor = 0.3f * height_factor + 0.7f * depth_factor;
        std::cout << "scale_factor: " << scale_factor << std::endl;
        std::cout << "---------------------------------- " << std::endl;

        // scale_factor = clamp(scale_factor, 0.1f, 1.0f);

        // 计算投影点位置
        float diagonal_length = cv::norm(diagonal_vec);
        float proj_length = diagonal_length * scale_factor;
        cv::Point2f proj_on_diagonal =
            diagonal_start + diagonal_vec * (proj_length / diagonal_length);

        // 计算椭圆参数
        float ellipse_width = box.right - box.left;
        int ellipse_height = ellipse_width * scale_factor;
        float angle =
            std::atan2(plane_tr.y - plane_tl.y, plane_tr.x - plane_tl.x) * 180 /
            CV_PI;
        cv::Point2f ellipse_center(box_center_2d.x, proj_on_diagonal.y);

        // 确保椭圆在检测框下方
        if (ellipse_center.y < box.bottom + roiOffsetY) {
          ellipse_center.y = box.bottom + roiOffsetY;
          float flattened_height = ellipse_height * 0.5f;
          if (ellipse_center.y - flattened_height / 2 <
              box.bottom + roiOffsetY) {
            ellipse_center.y = box.bottom + roiOffsetY + flattened_height / 2;
          }
          cv::Mat overlay;
          img.copyTo(overlay);
          cv::ellipse(overlay, ellipse_center,
                      cv::Size(ellipse_width / 2, flattened_height / 2), angle,
                      0, 360, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
          cv::addWeighted(overlay, 0.2, img, 0.8, 0, img);
        } else {
          cv::Mat overlay;
          img.copyTo(overlay);
          cv::ellipse(overlay, ellipse_center,
                      cv::Size(ellipse_width / 2, ellipse_height / 2), angle, 0,
                      360, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
          cv::addWeighted(overlay, 0.2, img, 0.8, 0, img);
        }
      }
    }
  }
  // -----------------------------------绘制水平红色虚线-----------------------------------
  if (tooth_detection_config.need_digging_start_location_) { // 静止虚线
    cv::Point line_start(1260, 1360);
    int line_length = 100;            // 虚线总长度
    int dash_length = 10;             // 单段虚线长度
    int gap_length = 5;               // 间隔长度
    cv::Scalar line_color(0, 0, 255); // BGR红色

    // 分段绘制
    int drawn_length = 0;
    while (drawn_length < line_length) {
      // 计算当前段终点
      cv::Point line_end =
          line_start +
          cv::Point(std::min(dash_length, line_length - drawn_length), 0);

      // 绘制实线部分
      if (line_start.x < img.cols && line_end.x < img.cols) {
        cv::line(img, line_start, line_end, line_color, 2, cv::LINE_AA);
      }

      // 更新位置
      drawn_length += (dash_length + gap_length);
      line_start.x += (dash_length + gap_length);
    }
  }
  return img;
}

// 全局或类成员变量来存储上一次追加的文件名
std::string lastRecordedFilename = "";
void save(const std::vector<std::string> &classNames,
          const std::string &savePath,
          ImageProcessingConfig &image_processing_config,
          ToothDetectionConfig &tooth_detection_config,
          PedestrianDetectionConfig &pedestrian_detection_config,
          InputConfig &input_config, ROIConfig &roi_config)

{
  // std::cout << "save函数正在运行!!! " << std::endl;
  cv::Scalar color = cv::Scalar(0, 255, 0);
  cv::Point bbox_points[1][4];
  const cv::Point *bbox_point0[1] = {bbox_points[0]};
  int num_points[] = {4};
  std::string miss_teeth = "";
  for (size_t bi = 0; bi < image_processing_config.imgs_batch.size(); bi++) {
    // 确定每张图像的ROI偏移
    int roiOffsetX = 0, roiOffsetY = 0;

    roiOffsetX = roi_config.roi_regions[input_config.idx][0];
    roiOffsetY = roi_config.roi_regions[input_config.idx][1];
    cv::rectangle(image_processing_config.imgs_batch[bi],
                  cv::Point(roi_config.roi_regions[input_config.idx][0],
                            roi_config.roi_regions[input_config.idx][1]),
                  cv::Point(roi_config.roi_regions[input_config.idx][2],
                            roi_config.roi_regions[input_config.idx][3]),
                  cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    if (!image_processing_config.bboxes.empty()) {
      for (auto &box : image_processing_config.bboxes[bi]) {
        if (classNames.size() == 15) {
          color = yolo_utils::Colors::colors_universal[box.label];
        }
        if (tooth_detection_config.tooth_state.count(box.label) ||
            box.label == 0) {
          // 行人检测框
          if (box.label == 0) { // 行人检测框
            cv::rectangle(
                image_processing_config.imgs_batch[bi],
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY),
                cv::Point(box.right + roiOffsetX, box.bottom + roiOffsetY),
                color, 2, cv::LINE_AA);
          } else if (image_processing_config.need_show_detection_info_ &&
                     !tooth_detection_config.show_tooth_state_name
                          .empty()) { // 铲斗检测框
            cv::rectangle(
                image_processing_config.imgs_batch[bi],
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY),
                cv::Point(box.right + roiOffsetX, box.bottom + roiOffsetY),
                color, 2, cv::LINE_AA);
            // 绘制标签和置信度
            // cv::String det_info = classNames[box.label] + " " +
            // cv::format("%.2f", box.confidence);
            cv::String det_info = tooth_detection_config.show_tooth_state_name +
                                  " " + cv::format("%.2f", box.confidence);
            bbox_points[0][0] =
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY);
            bbox_points[0][1] =
                cv::Point(box.left + det_info.size() * 18 + roiOffsetX,
                          box.top + roiOffsetY);
            bbox_points[0][2] =
                cv::Point(box.left + det_info.size() * 18 + roiOffsetX,
                          box.top + roiOffsetY - 23);
            bbox_points[0][3] =
                cv::Point(box.left + roiOffsetX, box.top + roiOffsetY - 23);
            cv::fillPoly(image_processing_config.imgs_batch[bi], bbox_point0,
                         num_points, 1, color);
            cv::putText(image_processing_config.imgs_batch[bi], det_info,
                        bbox_points[0][0], cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
          }
        }
        if (tooth_detection_config.teeth_and_root.count(box.label)) {
          cv::Mat create_alpha = cv::Mat::zeros(
              image_processing_config.imgs_batch[bi].rows,
              image_processing_config.imgs_batch[bi].cols, CV_8UC3);
          create_alpha.setTo(color);
          cv::Point pt1(box.left + roiOffsetX, box.top + roiOffsetY);
          cv::Point pt2(box.right + roiOffsetX, box.bottom + roiOffsetY);
          pt1.x = std::max(0, pt1.x);
          pt1.y = std::max(0, pt1.y);
          pt2.x = std::min(image_processing_config.imgs_batch[bi].cols, pt2.x);
          pt2.y = std::min(image_processing_config.imgs_batch[bi].rows, pt2.y);
          cv::Mat img_add;
          cv::addWeighted(
              image_processing_config.imgs_batch[bi](cv::Rect(pt1, pt2)), 0.7,
              create_alpha(cv::Rect(pt1, pt2)), 0.3, 0, img_add);
          img_add.copyTo(
              image_processing_config.imgs_batch[bi](cv::Rect(pt1, pt2)));
        }

        if (!box.land_marks.empty()) {
          for (auto &pt : box.land_marks) {
            cv::circle(image_processing_config.imgs_batch[bi], pt, 1,
                       cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
          }
        }
      }
    }

    // 构建文件名
    std::string timestamp = getCurrentTimestamp();
    std::string filename = savePath + (savePath.back() == '/' ? "" : "/") +
                           miss_teeth + "_" + timestamp + ".jpg";

    // 保存图像
    cv::imwrite(filename, image_processing_config.imgs_batch[bi]);

    // 检查文件名是否与上一次追加的相同
    if (filename != lastRecordedFilename) {
      // 打开结果文件追加结果
      std::string currentDate = getCurrentDate(); // 按照日期
      std::string resultFile = savePath + (savePath.back() == '/' ? "" : "/") +
                               currentDate + "_miss.txt";
      std::ofstream outfile(resultFile, std::ios::app);
      if (outfile.is_open()) {
        outfile << filename << std::endl;
        outfile.close();
        // 更新上一次追加的文件名
        lastRecordedFilename = filename;
      } else {
        std::cout << "Unable to open file" << std::endl;
      }
    }
  }
}

yolo_utils::HostTimer::HostTimer() { t1 = std::chrono::steady_clock::now(); }

float yolo_utils::HostTimer::getUsedTime() {
  t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  return (1000 * time_used.count()); // ms
}

yolo_utils::HostTimer::~HostTimer() {}

yolo_utils::DeviceTimer::DeviceTimer() {
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
}

float yolo_utils::DeviceTimer::getUsedTime() {
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float total_time;
  cudaEventElapsedTime(&total_time, start, end);
  return total_time;
}

yolo_utils::DeviceTimer::DeviceTimer(cudaStream_t stream) {
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
}

float yolo_utils::DeviceTimer::getUsedTime(cudaStream_t stream) {
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float total_time;
  cudaEventElapsedTime(&total_time, start, end);
  return total_time;
}

yolo_utils::DeviceTimer::~DeviceTimer() {
  cudaEventDestroy(start);
  cudaEventDestroy(end);
}
