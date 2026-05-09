#include "yolo_model.h"
#include <algorithm>

YOLOModel::YOLOModel(ImageProcessingConfig &image_processing_config,
                     InputConfig &input_config, ModelConfig &model_config,
                     ROIConfig &roi_config) {
  sample::gLogInfo << "Loading YOLO engine..." << std::endl;
  // 检查模型路径是否为空
  if (model_config.model_paths[input_config.idx].empty()) {
    sample::gLogError << "Model path is empty!" << std::endl;
    throw std::runtime_error("Model path is empty!");
  }

  // 加载模型
  trt_file = yolo_utils::loadModel(model_config.model_paths[input_config.idx]);
  if (trt_file.empty()) {
    sample::gLogError << "Failed to load model from path: "
                      << model_config.model_paths[input_config.idx]
                      << std::endl;
    throw std::runtime_error("Failed to load model!");
  }
  // sample::gLogInfo << "Load YOLO model file successfully!" << std::endl;

  // 初始化参数
  setParameters(param, roi_config, input_config, model_config);
  // sample::gLogInfo << "Initialize YOLO parameters successfully!" <<
  // std::endl;

  // 设置输入流，并初始化一些参数
  setInputStream(image_processing_config, input_config, param);
  // sample::gLogInfo << "Set input stream successfully!" << std::endl;

  // 使用智能指针来动态分配YOLOV11对象
  yolo =
      std::make_unique<YOLOV11>(param); // 使用std::unique_ptr来管理YOLOV11对象
  // yolo = std::make_shared<YOLOV11>(param); //
  // 使用std::make_shared来管理YOLOV11对象

  // sample::gLogInfo << "Initialize YOLO successfully!" << std::endl;

  // 初始化模型
  if (!yolo->init(trt_file)) // 使用指针访问成员函数
  {
    sample::gLogError << "Error initializing the engine!" << std::endl;
    throw std::runtime_error("Error initializing the engine!");
  }
  sample::gLogInfo << "Initialize YOLO engine successfully." << std::endl;
  // 预留内存
  image_processing_config.imgs_batch.reserve(param.batch_size);
  image_processing_config.crop_batch.reserve(param.batch_size);

  initialized = true; // 标记初始化完成
}

YOLOModel::~YOLOModel() {
  if (initialized) {
    // // 同步所有 CUDA 操作
    // cudaDeviceSynchronize();

    // // 确保 unique_ptr 管理的对象已释放
    // if (yolo)
    // {
    //     yolo.reset();
    // }

    // // 清理其他资源
    // if (capture.isOpened())
    // {
    //     capture.release();
    // }
    // trt_file.clear();

    // 标记为未初始化
    initialized = false;

    sample::gLogInfo << "YOLO model has been released." << std::endl;
  }
}

// 设置输入流
void YOLOModel::setParameters(InitParameter &initParameters,
                              ROIConfig &roi_config, InputConfig &input_config,
                              ModelConfig &model_config) {
  initParameters.class_names = yolo_utils::dataSets::universal_detect;
  initParameters.num_class = initParameters.class_names.size();
  // 二次检测模型（set_params_times 第二次进入）通常用于 crop/ROI 细分类，
  // 多路共享 batch 反而浪费显存，固定 batch=1。
  const bool is_second_model = (model_config.need_second_detection_ &&
                                (model_config.set_params_times + 1 == 2));

  // 共享模型多路场景：主模型尽量使用 batch 推理提升总吞吐（需要 TRT
  // 引擎支持动态 batch） 其他模式保持 batch=1，降低显存占用与延迟。
  if (use_nitros_sub_ && !is_second_model &&
      input_config.use_ros2_component_container &&
      input_config.process_method ==
          ProcessMethod::SHARED_MODEL_THREAD_MULTI_SOURCES) {
    const size_t n =
        input_config.camera_ids.empty() ? 1 : input_config.camera_ids.size();
    initParameters.batch_size = std::max<size_t>(1, n);
    initParameters.dynamic_batch = true;
  } else {
    initParameters.batch_size = 1;
  }
  initParameters.dst_h = 640;
  initParameters.dst_w = 640;
  initParameters.input_output_names = {"images", "output0"};
  initParameters.conf_thresh = 0.25f; // 置信度阈值
  initParameters.iou_thresh = 0.45f;  // NMS IOU阈值
  initParameters.save_path = "../../yolov11/save";
  initParameters.src_h = roi_config.roi_regions[input_config.idx][3] -
                         roi_config.roi_regions[input_config.idx][1];
  initParameters.src_w = roi_config.roi_regions[input_config.idx][2] -
                         roi_config.roi_regions[input_config.idx][0];

  // 设置二次检测时的输入流参数
  if (++model_config.set_params_times == 2 &&
      model_config.need_second_detection_) {
    initParameters.src_h = 640;
    initParameters.src_w = 640;
  }
}

// 获取param
InitParameter &YOLOModel::getParam() {
  if (!initialized) {
    throw std::runtime_error("YOLOModel is not initialized!");
  }
  return param;
}

// 获取YOLO对象
YOLOV11 &YOLOModel::getYolo() {
  if (!initialized) {
    throw std::runtime_error("YOLOModel is not initialized!");
  }
  return *yolo; // 解引用智能指针返回对象
}
