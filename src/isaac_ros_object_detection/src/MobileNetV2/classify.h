#ifndef TENSORRT_INFER_H
#define TENSORRT_INFER_H

#include "perception/config.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <logger.h>
#include <parserOnnxConfig.h>

using namespace nvinfer1;

class TensorRTInference {
public:
  TensorRTInference(ModelConfig &model_config);
  ~TensorRTInference();

  void predict(const cv::Mat &image, std::vector<float> &output);
  void predictFromGxfNv12GpuRoi(const void *nv12_gpu, int src_width,
                                int src_height, int y_stride, int uv_stride,
                                int roi_x, int roi_y, int roi_w, int roi_h,
                                cudaEvent_t ready_event,
                                std::vector<float> &output);
  bool isReady() const;

private:
  IRuntime *runtime_;
  ICudaEngine *engine_;
  IExecutionContext *context_;
  // std::unique_ptr<nvinfer1::IRuntime> runtime_;
  // std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  // std::unique_ptr<nvinfer1::IExecutionContext> context_;
  float *input_buffer_;
  float *output_buffer_;
  cudaStream_t stream_;
  int input_size_;
  int output_size_;
  int input_h_{224};
  int input_w_{224};
  unsigned char *roi_bgr_u8_{nullptr};
  float *tmp_bgr_f32_{nullptr};
  float *tmp_rgb_f32_{nullptr};
  float *tmp_norm_f32_{nullptr};
  InitParameter norm_param_{};
};

#endif // TENSORRT_INFER_H
