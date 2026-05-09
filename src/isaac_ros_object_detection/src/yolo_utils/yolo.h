#pragma once
#include "kernel_function.h"
#include "perception/config.h"

namespace yolo {
class YOLO {
public:
  YOLO() = default;
  YOLO(const InitParameter &param);
  ~YOLO();

public:
  virtual bool init(const std::vector<unsigned char> &trtFile);
  virtual void check();
  virtual void copy(const std::vector<cv::Mat> &imgsBatch);
  // 直接从 GPU 的 GXF NV12 缓冲填充 YOLO 输入（避免 NV12->BGR->CPU->H2D）
  // 注意：该接口假设 width/height 与 m_param.src_w/src_h 一致（即 ROI
  // 已在上游保证）
  virtual void copyFromGxfNv12Gpu(const void *nv12_gpu, int width, int height,
                                  int y_stride, int uv_stride,
                                  cudaEvent_t ready_event = nullptr,
                                  int batch_index = 0);
  virtual void copyFromGxfNv12GpuRoiResize(const void *nv12_gpu, int src_width,
                                           int src_height, int y_stride,
                                           int uv_stride, int roi_x, int roi_y,
                                           int roi_w, int roi_h,
                                           cudaEvent_t ready_event = nullptr,
                                           int batch_index = 0);
  // 无需 cv::Mat 的预处理入口（基于已填充的 m_input_src_device）
  virtual void preprocess();
  virtual void preprocess(const std::vector<cv::Mat> &imgsBatch);
  virtual bool infer();
  virtual void postprocess(const std::vector<cv::Mat> &imgsBatch);
  // 无需 cv::Mat 的后处理入口（batch_size 用于决定解析多少张图）
  virtual void postprocess(int batch_size);
  virtual void reset();

public:
  std::vector<std::vector<Box>> getObjectss() const;

protected:
  std::unique_ptr<nvinfer1::IRuntime> m_runtime;
  // nvinfer1::IRuntime *m_runtime;
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  // nvinfer1::ICudaEngine *m_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;
  // nvinfer1::IExecutionContext *m_context;

protected:
  InitParameter m_param;
  nvinfer1::Dims m_output_dims;
  int m_output_area;
  int m_total_objects;
  std::vector<std::vector<Box>> m_objectss;
  yolo_utils::AffineMat m_dst2src;

  // input
  unsigned char *m_input_src_device;
  float *m_input_resize_device;
  float *m_input_rgb_device;
  float *m_input_norm_device;
  float *m_input_hwc_device;
  // output
  float *m_output_src_device;
  float *m_output_objects_device;
  float *m_output_objects_host;
  int m_output_objects_width;
  int *m_output_idx_device;
  float *m_output_conf_device;

  // 让预处理/推理/后处理都跑在同一个非默认 stream，避免 legacy default stream
  // 与多源其他 stream 的隐式同步导致“几百毫秒抖动”。
  cudaStream_t m_stream_{nullptr};
};
} // namespace yolo
