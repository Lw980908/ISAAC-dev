#include "classify.h"
#include "../perception/gxf_nv12_roi_resize.h"
#include "../yolo_utils/kernel_function.h"
#include "../yolo_utils/yolo_utils.h"
#include "yolo_utils/trt_compat.h" // TensorRT 8.x/10.x 兼容性
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace nvinfer1;

TensorRTInference::TensorRTInference(ModelConfig &model_config) {
  runtime_ = nullptr;
  engine_ = nullptr;
  context_ = nullptr;
  input_buffer_ = nullptr;
  output_buffer_ = nullptr;
  stream_ = nullptr;
  input_size_ = 0;
  output_size_ = 0;
  if (model_config.need_classifier_) {
    const auto logLoadedLib = [](const char *needle) {
      std::ifstream maps("/proc/self/maps");
      if (!maps) {
        return;
      }
      std::string line;
      while (std::getline(maps, line)) {
        if (line.find(needle) != std::string::npos) {
          sample::gLogInfo << needle << " loaded: " << line << std::endl;
          return;
        }
      }
    };

    logLoadedLib("libnvinfer.so");
    logLoadedLib("libnvinfer_plugin.so");
    logLoadedLib("libnvonnxparser.so");

    sample::gLogInfo << "TensorRT headers: " << NV_TENSORRT_MAJOR << "."
                     << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
                     << std::endl;
    sample::gLogInfo << "Loading MobileNetV2 engine: "
                     << model_config.classify_model_path << std::endl;
    std::ifstream file(model_config.classify_model_path, std::ios::binary);
    if (!file) {
      std::cerr << "Failed to load engine: " << model_config.classify_model_path
                << std::endl;
      return;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trt_model_stream = new char[size];
    file.read(trt_model_stream, size);
    if (!file) {
      std::cerr << "Read engine file failed: "
                << model_config.classify_model_path
                << " read=" << static_cast<size_t>(file.gcount())
                << " expected=" << size << std::endl;
      delete[] trt_model_stream;
      return;
    }
    file.close();

    sample::gLogInfo << "Engine file bytes[0..15]: ";
    for (int i = 0; i < 16 && i < static_cast<int>(size); ++i) {
      sample::gLogInfo << std::hex << std::setw(2) << std::setfill('0')
                       << (static_cast<unsigned>(
                              static_cast<unsigned char>(trt_model_stream[i])))
                       << " ";
    }
    sample::gLogInfo << std::dec << std::endl;

    runtime_ = createInferRuntime(sample::gLogger.getTRTLogger());
    if (!runtime_) {
      std::cerr << "创建 Runtime 失败！" << std::endl;
      return;
    }

    // TensorRT 8.x/10.x 兼容
    engine_ = TRT_DESERIALIZE_ENGINE(runtime_, trt_model_stream, size);
    if (!engine_) {
      std::cerr << "分类模型引擎反序列化失败！" << std::endl;
      delete[] trt_model_stream;
      return;
    }
    delete[] trt_model_stream;
    context_ = engine_->createExecutionContext();
    if (!context_) {
      std::cerr << "分类模型上下文创建失败！" << std::endl;
      return;
    }

    // TensorRT 8.x/10.x 兼容：获取输入维度
    auto input_dims = TRT_GET_BINDING_DIMS(engine_, 0);
    input_size_ = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
      input_size_ *= input_dims.d[i];
    }
    if (input_dims.nbDims == 4) {
      input_h_ = input_dims.d[2] > 0 ? input_dims.d[2] : 224;
      input_w_ = input_dims.d[3] > 0 ? input_dims.d[3] : 224;
    }
    cudaMalloc(reinterpret_cast<void **>(&input_buffer_),
               input_size_ * sizeof(float));

    // TensorRT 8.x/10.x 兼容：获取输出维度
    auto output_dims = TRT_GET_BINDING_DIMS(engine_, 1);
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
      output_size_ *= output_dims.d[i];
    }
    cudaMalloc(reinterpret_cast<void **>(&output_buffer_),
               output_size_ * sizeof(float));

    cudaStreamCreate(&stream_);

    norm_param_.scale = 255.f;
    norm_param_.means[0] = 0.485f;
    norm_param_.means[1] = 0.485f;
    norm_param_.means[2] = 0.485f;
    norm_param_.stds[0] = 0.229f;
    norm_param_.stds[1] = 0.229f;
    norm_param_.stds[2] = 0.229f;

    const size_t roi_pixels =
        static_cast<size_t>(input_h_) * static_cast<size_t>(input_w_);
    cudaMalloc(reinterpret_cast<void **>(&roi_bgr_u8_), roi_pixels * 3);
    cudaMalloc(reinterpret_cast<void **>(&tmp_bgr_f32_),
               roi_pixels * 3 * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&tmp_rgb_f32_),
               roi_pixels * 3 * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&tmp_norm_f32_),
               roi_pixels * 3 * sizeof(float));

    sample::gLogInfo << "Initialize MobileNetV2 engine successfully."
                     << std::endl;
  } else {
    sample::gLogInfo << "No need to classify." << std::endl;
  }
}

TensorRTInference::~TensorRTInference() {
  if (engine_) // 仅在需要分类时才释放资源，即模型不为空
  {
    // TensorRT 8.x/10.x 兼容：使用 TrtDeleter 处理资源释放
#if TRT_VERSION_10_OR_LATER
    // TensorRT 10.x: 使用 delete 释放资源
    delete context_;
    delete engine_;
    delete runtime_;
#else
    // TensorRT 8.x: 使用 destroy() 方法
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
#endif
    cudaStreamDestroy(stream_);
    cudaFree(input_buffer_);
    cudaFree(output_buffer_);
    cudaFree(roi_bgr_u8_);
    cudaFree(tmp_bgr_f32_);
    cudaFree(tmp_rgb_f32_);
    cudaFree(tmp_norm_f32_);
    sample::gLogInfo << "MobileNetV2 model has been released." << std::endl;
  }
}

void TensorRTInference::predict(const cv::Mat &image,
                                std::vector<float> &output) {
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(input_w_, input_h_));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  float *host_input = new float[input_size_];
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < input_h_; h++) {
      for (int w = 0; w < input_w_; w++) {
        host_input[c * input_h_ * input_w_ + h * input_w_ + w] =
            (resized.at<cv::Vec3b>(h, w)[c] / 255.0 - 0.485) / 0.229;
      }
    }
  }

  cudaMemcpyAsync(input_buffer_, host_input, input_size_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);
  delete[] host_input;

  void *bindings[2];
  bindings[0] = input_buffer_;
  bindings[1] = output_buffer_;

  // TensorRT 8.x/10.x 兼容：executeV2 vs enqueueV3
#if TRT_VERSION_10_OR_LATER
  // TensorRT 10.x: 使用 setTensorAddress + enqueueV3
  const char *input_name = engine_->getIOTensorName(0);
  const char *output_name = engine_->getIOTensorName(1);
  context_->setTensorAddress(input_name, input_buffer_);
  context_->setTensorAddress(output_name, output_buffer_);
  context_->enqueueV3(stream_);
#else
  // TensorRT 8.x: 使用 executeV2
  context_->executeV2(bindings);
#endif

  output.resize(output_size_);
  cudaMemcpyAsync(output.data(), output_buffer_, output_size_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  float max_val = *std::max_element(output.begin(), output.end());
  float sum_exp = 0.0f;
  for (float &val : output) {
    val = exp(val - max_val);
    sum_exp += val;
  }
  for (float &val : output) {
    val /= sum_exp;
  }
}

void TensorRTInference::predictFromGxfNv12GpuRoi(
    const void *nv12_gpu, int src_width, int src_height, int y_stride,
    int uv_stride, int roi_x, int roi_y, int roi_w, int roi_h,
    cudaEvent_t ready_event, std::vector<float> &output) {
  if (!engine_ || !context_ || !nv12_gpu) {
    output.clear();
    return;
  }

  roi_x = std::max(0, std::min(roi_x, src_width - 1));
  roi_y = std::max(0, std::min(roi_y, src_height - 1));
  roi_w = std::max(1, std::min(roi_w, src_width - roi_x));
  roi_h = std::max(1, std::min(roi_h, src_height - roi_y));

  if (ready_event) {
    cudaStreamWaitEvent(stream_, ready_event, 0);
  }

  const int dst_bgr_stride = input_w_ * 3;
  ConvertGxfNV12RoiResizeToBGR(roi_bgr_u8_, input_w_, input_h_, dst_bgr_stride,
                               nv12_gpu, src_width, src_height, y_stride,
                               uv_stride, roi_x, roi_y, roi_w, roi_h, stream_);

  yolo_utils::AffineMat identity;
  identity.v0 = 1.f;
  identity.v1 = 0.f;
  identity.v2 = 0.f;
  identity.v3 = 0.f;
  identity.v4 = 1.f;
  identity.v5 = 0.f;

  resizeDevice(1, roi_bgr_u8_, input_w_, input_h_, tmp_bgr_f32_, input_w_,
               input_h_, 0.f, identity, stream_);
  bgr2rgbDevice(1, tmp_bgr_f32_, input_w_, input_h_, tmp_rgb_f32_, input_w_,
                input_h_, stream_);
  normDevice(1, tmp_rgb_f32_, input_w_, input_h_, tmp_norm_f32_, input_w_,
             input_h_, norm_param_, stream_);
  hwc2chwDevice(1, tmp_norm_f32_, input_w_, input_h_, input_buffer_, input_w_,
                input_h_, stream_);

#if TRT_VERSION_10_OR_LATER
  const char *input_name = engine_->getIOTensorName(0);
  const char *output_name = engine_->getIOTensorName(1);
  context_->setTensorAddress(input_name, input_buffer_);
  context_->setTensorAddress(output_name, output_buffer_);
  context_->enqueueV3(stream_);
#else
  void *bindings[2];
  bindings[0] = input_buffer_;
  bindings[1] = output_buffer_;
  context_->executeV2(bindings);
#endif

  output.resize(output_size_);
  cudaMemcpyAsync(output.data(), output_buffer_, output_size_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  float max_val = *std::max_element(output.begin(), output.end());
  float sum_exp = 0.0f;
  for (float &val : output) {
    val = exp(val - max_val);
    sum_exp += val;
  }
  for (float &val : output) {
    val /= sum_exp;
  }
}

bool TensorRTInference::isReady() const {
  return engine_ && context_ && input_buffer_ && output_buffer_ && stream_ &&
         roi_bgr_u8_ && tmp_bgr_f32_ && tmp_rgb_f32_ && tmp_norm_f32_;
}
