#include "yolo.h"
#include "../perception/gxf_nv12_convert.h"
#include "../perception/gxf_nv12_roi_resize.h"
#include "trt_compat.h" // TensorRT 8.x/10.x 兼容性

yolo::YOLO::YOLO(const InitParameter &param) : m_param(param) {
  // 修改所有CHECK(cudaMalloc)为带错误检查的版本
  auto cudaCheck = [](cudaError_t code) {
    if (code != cudaSuccess) {
      throw std::runtime_error("CUDA内存分配失败: " +
                               std::string(cudaGetErrorString(code)));
    }
  };
  // input
  m_input_src_device = nullptr;
  m_input_resize_device = nullptr;
  m_input_rgb_device = nullptr;
  m_input_norm_device = nullptr;
  m_input_hwc_device = nullptr;
  CHECK(cudaMalloc(&m_input_src_device, param.batch_size * 3 * param.src_h *
                                            param.src_w *
                                            sizeof(unsigned char)));
  CHECK(cudaMalloc(&m_input_resize_device, param.batch_size * 3 * param.dst_h *
                                               param.dst_w * sizeof(float)));
  CHECK(cudaMalloc(&m_input_rgb_device, param.batch_size * 3 * param.dst_h *
                                            param.dst_w * sizeof(float)));
  CHECK(cudaMalloc(&m_input_norm_device, param.batch_size * 3 * param.dst_h *
                                             param.dst_w * sizeof(float)));
  CHECK(cudaMalloc(&m_input_hwc_device, param.batch_size * 3 * param.dst_h *
                                            param.dst_w * sizeof(float)));

  // dedicated stream for this YOLO instance
  CHECK(cudaStreamCreate(&m_stream_));

  // output
  m_output_src_device = nullptr;
  m_output_objects_device = nullptr;
  m_output_objects_host = nullptr;
  m_output_objects_width = 7;
  m_output_idx_device = nullptr;
  m_output_conf_device = nullptr;
  int output_objects_size =
      param.batch_size * (1 + param.topK * m_output_objects_width); // 1: count
  CHECK(cudaMalloc(&m_output_objects_device,
                   output_objects_size * sizeof(float)));
  CHECK(cudaMalloc(&m_output_idx_device,
                   m_param.batch_size * m_param.topK * sizeof(int)));
  CHECK(cudaMalloc(&m_output_conf_device,
                   m_param.batch_size * m_param.topK * sizeof(float)));
  // pinned host memory for true async D2H
  CHECK(cudaHostAlloc(&m_output_objects_host,
                      output_objects_size * sizeof(float),
                      cudaHostAllocDefault));
  m_objectss.resize(param.batch_size);
}

yolo::YOLO::~YOLO() {
  // 1. 只等待本实例 stream 上的任务完成，避免全局同步抖动
  if (m_stream_) {
    cudaStreamSynchronize(m_stream_);
  }

  // // 2. 销毁TensorRT对象
  // if (m_context)
  // {
  //     m_context->destroy();
  //     m_context = nullptr;
  // } // 先销毁上下文
  // if (m_engine)
  // {
  //     m_engine->destroy();
  //     m_engine = nullptr;
  // } // 再销毁引擎
  // if (m_runtime)
  // {
  //     m_runtime->destroy();
  //     m_runtime = nullptr;
  // } // 最后销毁运行时

  // 智能指针自动销毁TensorRT对象（顺序：context → engine → runtime）
  m_context.reset();
  m_engine.reset();
  m_runtime.reset();

  // 3. 释放所有CUDA内存（input -> output顺序非必需）
  // input
  if (m_input_src_device) {
    // CHECK(cudaFree(m_input_src_device));
    cudaFree(m_input_src_device);
    m_input_src_device = nullptr;
  }
  if (m_input_resize_device) {
    // CHECK(cudaFree(m_input_resize_device));
    cudaFree(m_input_resize_device);
    m_input_resize_device = nullptr;
  }
  if (m_input_rgb_device) {
    // CHECK(cudaFree(m_input_rgb_device));
    cudaFree(m_input_rgb_device);
    m_input_rgb_device = nullptr;
  }
  if (m_input_norm_device) {
    // CHECK(cudaFree(m_input_norm_device));
    cudaFree(m_input_norm_device);
    m_input_norm_device = nullptr;
  }
  if (m_input_hwc_device) {
    // CHECK(cudaFree(m_input_hwc_device));
    cudaFree(m_input_hwc_device);
    m_input_hwc_device = nullptr;
  }

  // output
  if (m_output_src_device) {
    // CHECK(cudaFree(m_output_src_device));
    cudaFree(m_output_src_device);
    m_output_src_device = nullptr;
  }
  if (m_output_objects_device) {
    // CHECK(cudaFree(m_output_objects_device));
    cudaFree(m_output_objects_device);
    m_output_objects_device = nullptr;
  }
  if (m_output_idx_device) {
    // CHECK(cudaFree(m_output_idx_device));
    cudaFree(m_output_idx_device);
    m_output_idx_device = nullptr;
  }
  if (m_output_conf_device) {
    // CHECK(cudaFree(m_output_conf_device));
    cudaFree(m_output_conf_device);
    m_output_conf_device = nullptr;
  }

  // 4. 释放主机内存
  if (m_output_objects_host) {
    cudaFreeHost(m_output_objects_host);
    m_output_objects_host = nullptr;
  }

  if (m_stream_) {
    cudaStreamDestroy(m_stream_);
    m_stream_ = nullptr;
  }
}

bool yolo::YOLO::init(const std::vector<unsigned char> &trtFile) {
  if (trtFile.empty()) {
    return false;
  }

  // 使用兼容性删除器的智能指针
  this->m_runtime = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
  if (m_runtime == nullptr) {
    return false;
  }

  // TensorRT 8.x/10.x 兼容：deserializeCudaEngine 参数不同
  this->m_engine = makeTrtShared<nvinfer1::ICudaEngine>(
      TRT_DESERIALIZE_ENGINE(m_runtime.get(), trtFile.data(), trtFile.size()));
  if (this->m_engine == nullptr) {
    return false;
  }

  this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(
      this->m_engine->createExecutionContext());
  if (this->m_context == nullptr) {
    return false;
  }

  if (m_param.dynamic_batch) // for some models only support static mutil-batch.
                             // eg: yolox
  {
    // TensorRT 8.x/10.x 兼容：setBindingDimensions vs setInputShape
    TRT_SET_INPUT_SHAPE(
        m_context.get(), m_engine.get(), 0,
        nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
  }
  // TensorRT 8.x/10.x 兼容：getBindingDimensions vs getTensorShape
  m_output_dims =
      TRT_CONTEXT_GET_BINDING_DIMS(m_context.get(), m_engine.get(), 1);
  m_total_objects = m_output_dims.d[1];
  if (m_output_dims.d[0] < static_cast<int>(m_param.batch_size)) {
    sample::gLogError << "TensorRT engine max batch (" << m_output_dims.d[0]
                      << ") < requested batch (" << m_param.batch_size << ")"
                      << std::endl;
    return false;
  }
  m_output_area = 1;
  for (int i = 1; i < m_output_dims.nbDims; i++) {
    if (m_output_dims.d[i] != 0) {
      m_output_area *= m_output_dims.d[i];
    }
  }
  CHECK(cudaMalloc(&m_output_src_device,
                   m_param.batch_size * m_output_area * sizeof(float)));
  float a = float(m_param.dst_h) / m_param.src_h;
  float b = float(m_param.dst_w) / m_param.src_w;
  float scale = a < b ? a : b;
  cv::Mat src2dst =
      (cv::Mat_<float>(2, 3) << scale, 0.f,
       (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5, 0.f, scale,
       (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
  cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
  cv::invertAffineTransform(src2dst, dst2src);
  m_dst2src.v0 = dst2src.ptr<float>(0)[0];
  m_dst2src.v1 = dst2src.ptr<float>(0)[1];
  m_dst2src.v2 = dst2src.ptr<float>(0)[2];
  m_dst2src.v3 = dst2src.ptr<float>(1)[0];
  m_dst2src.v4 = dst2src.ptr<float>(1)[1];
  m_dst2src.v5 = dst2src.ptr<float>(1)[2];
  return true;
}

void yolo::YOLO::check() {
  int idx;
  nvinfer1::Dims dims;

  sample::gLogInfo << "the engine's info:" << std::endl;
  for (auto layer_name : m_param.input_output_names) {
    // TensorRT 8.x/10.x 兼容：getBindingIndex
    idx = TRT_GET_BINDING_INDEX(m_engine.get(), layer_name.c_str());
    dims = TRT_GET_BINDING_DIMS(m_engine.get(), idx);
    sample::gLogInfo << "idx = " << idx << ", " << layer_name << ": ";
    for (int i = 0; i < dims.nbDims; i++) {
      sample::gLogInfo << dims.d[i] << ", ";
    }
    sample::gLogInfo << std::endl;
  }
  sample::gLogInfo << "the context's info:" << std::endl;
  for (auto layer_name : m_param.input_output_names) {
    idx = TRT_GET_BINDING_INDEX(m_engine.get(), layer_name.c_str());
    dims = TRT_CONTEXT_GET_BINDING_DIMS(m_context.get(), m_engine.get(), idx);
    sample::gLogInfo << "idx = " << idx << ", " << layer_name << ": ";
    for (int i = 0; i < dims.nbDims; i++) {
      sample::gLogInfo << dims.d[i] << ", ";
    }
    sample::gLogInfo << std::endl;
  }
}
void yolo::YOLO::copy(const std::vector<cv::Mat> &imgsBatch) {
#if 0 
    cv::Mat img_fp32 = cv::Mat::zeros(imgsBatch[0].size(), CV_32FC3); // todo 
    cudaHostRegister(img_fp32.data, img_fp32.elemSize() * img_fp32.total(), cudaHostRegisterPortable);
    float* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        imgsBatch[i].convertTo(img_fp32, CV_32FC3);
        CHECK(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        pi += 3 * m_param.src_h * m_param.src_w;
    }
    cudaHostUnregister(img_fp32.data);
#endif

#if 0 // for Nvidia TX2
    cv::Mat img_fp32 = cv::Mat::zeros(imgsBatch[0].size(), CV_32FC3); // todo 
    float* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        std::vector<float> img_vec = std::vector<float>(imgsBatch[i].reshape(1, 1));
        imgsBatch[i].convertTo(img_fp32, CV_32FC3);
        CHECK(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        pi += 3 * m_param.src_h * m_param.src_w;
    }
#endif

  // update 20230302, faster.
  // 1. move uint8_to_float in cuda kernel function. For 8*3*1920*1080, cost
  // time 15ms -> 3.9ms
  // 2. Todo
  unsigned char *pi = m_input_src_device;
  for (size_t i = 0; i < imgsBatch.size(); i++) {
    CHECK(cudaMemcpyAsync(pi, imgsBatch[i].data,
                          sizeof(unsigned char) * 3 * m_param.src_h *
                              m_param.src_w,
                          cudaMemcpyHostToDevice, m_stream_));
    pi += 3 * m_param.src_h * m_param.src_w;
  }

#if 0 // cuda stream
    cudaStream_t streams[32];
    for (int i = 0; i < imgsBatch.size(); i++) 
    {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    unsigned char* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        CHECK(cudaMemcpyAsync(pi, imgsBatch[i].data, sizeof(unsigned char) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice, streams[i]));
        pi += 3 * m_param.src_h * m_param.src_w;
    }
    for (int i = 0; i < imgsBatch.size(); i++)
    {
        CHECK(cudaStreamSynchronize(streams[i]));
    }
#endif
}

void yolo::YOLO::copyFromGxfNv12Gpu(const void *nv12_gpu, int width, int height,
                                    int y_stride, int uv_stride,
                                    cudaEvent_t ready_event, int batch_index) {
  if (!nv12_gpu) {
    return;
  }
  if (!m_input_src_device) {
    throw std::runtime_error("YOLO input buffer is null");
  }
  if (batch_index < 0 ||
      static_cast<size_t>(batch_index) >= m_param.batch_size) {
    throw std::runtime_error("batch_index out of range");
  }
  // 强约束：共享模型场景下 ROI 应在上游保证一致；这里避免隐式裁剪导致错位
  if (width != m_param.src_w || height != m_param.src_h) {
    throw std::runtime_error("NV12 frame size mismatch: expect src_w/src_h");
  }

  if (ready_event) {
    CHECK(cudaStreamWaitEvent(m_stream_, ready_event, 0));
  }

  const int bgr_stride = width * 3;
  auto *dst_bgr =
      reinterpret_cast<uint8_t *>(m_input_src_device) +
      static_cast<size_t>(batch_index) * 3 * m_param.src_h * m_param.src_w;
  CHECK(ConvertGxfNV12ToBGR(dst_bgr, nv12_gpu, width, height, y_stride,
                            uv_stride, bgr_stride, m_stream_));
}

void yolo::YOLO::copyFromGxfNv12GpuRoiResize(const void *nv12_gpu,
                                             int src_width, int src_height,
                                             int y_stride, int uv_stride,
                                             int roi_x, int roi_y, int roi_w,
                                             int roi_h, cudaEvent_t ready_event,
                                             int batch_index) {
  if (!nv12_gpu) {
    return;
  }
  if (!m_input_src_device) {
    throw std::runtime_error("YOLO input buffer is null");
  }
  if (batch_index < 0 ||
      static_cast<size_t>(batch_index) >= m_param.batch_size) {
    throw std::runtime_error("batch_index out of range");
  }

  if (ready_event) {
    CHECK(cudaStreamWaitEvent(m_stream_, ready_event, 0));
  }

  const int dst_w = m_param.src_w;
  const int dst_h = m_param.src_h;
  const int bgr_stride = dst_w * 3;
  auto *dst_bgr = reinterpret_cast<uint8_t *>(m_input_src_device) +
                  static_cast<size_t>(batch_index) * 3 * dst_h * dst_w;

  CHECK(ConvertGxfNV12RoiResizeToBGR(
      dst_bgr, dst_w, dst_h, bgr_stride, nv12_gpu, src_width, src_height,
      y_stride, uv_stride, roi_x, roi_y, roi_w, roi_h, m_stream_));
}

void yolo::YOLO::preprocess() {
  resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w,
               m_param.src_h, m_input_resize_device, m_param.dst_w,
               m_param.dst_h, 114, m_dst2src, m_stream_);
  bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w,
                m_param.dst_h, m_input_rgb_device, m_param.dst_w, m_param.dst_h,
                m_stream_);
  normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w,
             m_param.dst_h, m_input_norm_device, m_param.dst_w, m_param.dst_h,
             m_param, m_stream_);
  hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w,
                m_param.dst_h, m_input_hwc_device, m_param.dst_w, m_param.dst_h,
                m_stream_);
}

void yolo::YOLO::preprocess(const std::vector<cv::Mat> &imgsBatch) {
  (void)imgsBatch;
  preprocess();
}

bool yolo::YOLO::infer() {
  float *bindings[] = {m_input_hwc_device, m_output_src_device};
  // TensorRT 8.x/10.x 兼容：executeV2 vs enqueueV3
  bool context = TRT_EXECUTE_V2_ASYNC(m_context.get(), m_engine.get(), bindings,
                                      m_stream_);
  return context;
}

void yolo::YOLO::postprocess(const std::vector<cv::Mat> &imgsBatch) {
  postprocess(static_cast<int>(imgsBatch.size()));
}

void yolo::YOLO::postprocess(int batch_size) {
  if (batch_size <= 0) {
    return;
  }
  batch_size = std::min(batch_size, static_cast<int>(m_param.batch_size));

  decodeDevice(m_param, m_output_src_device, 5 + m_param.num_class,
               m_total_objects, m_output_area, m_output_objects_device,
               m_output_objects_width, m_param.topK, m_stream_);

  // nmsv1(nms faster)
  nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width,
              m_param.topK, m_param.topK * m_output_objects_width + 1,
              m_stream_);

  // nmsv2(nms sort)
  // nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width,
  // m_param.topK, m_param.topK * m_output_objects_width + 1,
  // m_output_idx_device, m_output_conf_device);

  CHECK(cudaMemcpyAsync(m_output_objects_host, m_output_objects_device,
                        m_param.batch_size * sizeof(float) *
                            (1 + 7 * m_param.topK),
                        cudaMemcpyDeviceToHost, m_stream_));
  CHECK(cudaStreamSynchronize(m_stream_));
  for (int bi = 0; bi < batch_size; bi++) {
    int num_boxes =
        std::min((int)(m_output_objects_host +
                       bi * (m_param.topK * m_output_objects_width + 1))[0],
                 m_param.topK);
    for (size_t i = 0; i < num_boxes; i++) {
      float *ptr = m_output_objects_host +
                   bi * (m_param.topK * m_output_objects_width + 1) +
                   m_output_objects_width * i + 1;
      int keep_flag = ptr[6];
      if (keep_flag) {
        float x_lt =
            m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2;
        float y_lt =
            m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
        float x_rb =
            m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2;
        float y_rb =
            m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;
        m_objectss[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, ptr[4],
                                    (int)ptr[5]);
      }
    }
  }
}

std::vector<std::vector<Box>> yolo::YOLO::getObjectss() const {
  return this->m_objectss;
}

void yolo::YOLO::reset() {
  CHECK(cudaMemsetAsync(
      m_output_objects_device, 0,
      sizeof(float) * m_param.batch_size * (1 + 7 * m_param.topK), m_stream_));
  for (size_t bi = 0; bi < m_param.batch_size; bi++) {
    m_objectss[bi].clear();
  }
}
