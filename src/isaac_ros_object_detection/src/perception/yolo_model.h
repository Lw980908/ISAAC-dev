#pragma once
#include "config.h"
#include "yolo_utils/yolov11.h"

class YOLOModel
{
public:
    YOLOModel() = default; // 默认构造函数
    YOLOModel(ImageProcessingConfig &image_processing_config, InputConfig &input_config, ModelConfig &model_config, ROIConfig &roi_config);
    virtual ~YOLOModel();

    // 设置输入流
    void setParameters(InitParameter &initParameters, ROIConfig &roi_config, InputConfig &input_config, ModelConfig &model_config);
    // 获取param
    virtual InitParameter &getParam();
    // 获取YOLO对象
    virtual YOLOV11 &getYolo();

private:
    bool initialized = false;
    InputStream source;
    std::vector<unsigned char> trt_file;
    InitParameter param;
    std::unique_ptr<YOLOV11> yolo; // 使用unique_ptr来管理YOLOV11对象
    // std::shared_ptr<YOLOV11> yolo; // 使用shared_ptr来管理YOLOV11对象
};