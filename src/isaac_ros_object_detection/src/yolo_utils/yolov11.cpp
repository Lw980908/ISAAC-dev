#include "yolov11.h"
#include "decode_yolov11.h"
#include "trt_compat.h" // TensorRT 8.x/10.x 兼容性

YOLOV11::YOLOV11(const InitParameter &param) : yolo::YOLO(param) {}

YOLOV11::~YOLOV11() {
  if (m_output_src_transpose_device) {
    cudaFree(m_output_src_transpose_device);
    m_output_src_transpose_device = nullptr; // 避免双重释放
  }
}

bool YOLOV11::init(const std::vector<unsigned char> &trtFile) {
  if (trtFile.empty()) {
    return false;
  }
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

  if (m_param.dynamic_batch) {
    // TensorRT 8.x/10.x 兼容：setBindingDimensions vs setInputShape
    TRT_SET_INPUT_SHAPE(
        m_context.get(), m_engine.get(), 0,
        nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
  }
  // TensorRT 8.x/10.x 兼容：getBindingDimensions vs getTensorShape
  m_output_dims =
      TRT_CONTEXT_GET_BINDING_DIMS(m_context.get(), m_engine.get(), 1);
  m_total_objects = m_output_dims.d[2];
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
  CHECK(cudaMalloc(&m_output_src_transpose_device,
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

void YOLOV11::preprocess(const std::vector<cv::Mat> &imgsBatch) {
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

void YOLOV11::postprocess(const std::vector<cv::Mat> &imgsBatch) {
  postprocess(static_cast<int>(imgsBatch.size()));
}

void YOLOV11::postprocess(int batch_size) {
  if (batch_size <= 0) {
    return;
  }
  batch_size = std::min(batch_size, static_cast<int>(m_param.batch_size));

  yolov11::transposeDevice(
      m_param, m_output_src_device, m_total_objects, 4 + m_param.num_class,
      m_total_objects * (4 + m_param.num_class), m_output_src_transpose_device,
      4 + m_param.num_class, m_total_objects, m_stream_);
  yolov11::decodeDevice(m_param, m_output_src_transpose_device,
                        4 + m_param.num_class, m_total_objects, m_output_area,
                        m_output_objects_device, m_output_objects_width,
                        m_param.topK, m_stream_);
  // nms
  // nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width,
  // m_param.topK, m_param.topK * m_output_objects_width + 1);
  nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width,
              m_param.topK, m_param.topK * m_output_objects_width + 1,
              m_output_idx_device, m_output_conf_device, m_stream_);
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