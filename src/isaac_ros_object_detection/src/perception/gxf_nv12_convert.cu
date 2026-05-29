// GXF NV12 格式转换 CUDA kernel
// 用于 GXF NV12 格式与 BGR 之间的转换（绕过 VPI）

#include <cuda_runtime.h>
#include <stdint.h>

// ============== GXF NV12 <-> BGR 直接转换 ==============

// 辅助函数：将 float 限制在 [0, 255] 范围内
__device__ __forceinline__ uint8_t clamp_to_uint8(float val) {
  return static_cast<uint8_t>(fminf(fmaxf(val, 0.0f), 255.0f));
}

// CUDA kernel: GXF NV12 直接转换为 BGR（完全在 GPU 上）
// GXF NV12 格式：Y stride = width, UV stride = width * 2
__global__ void GxfNV12ToBGRKernel(uint8_t *__restrict__ bgr,
                                   const uint8_t *__restrict__ y_plane,
                                   const uint8_t *__restrict__ uv_plane,
                                   int width, int height, int y_stride,
                                   int uv_stride, int bgr_stride,
                                   int swap_uv) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // 读取 Y 值
  const float Y = static_cast<float>(y_plane[y * y_stride + x]);

  // 计算 UV 坐标（UV 平面是 Y 平面的一半分辨率）
  const int uv_x = x / 2;
  const int uv_y = y / 2;
  // GXF NV12: UV 交错存储，每对 UV 占 2 字节
  // uv_plane[uv_y * uv_stride + uv_x * 2] = U
  // uv_plane[uv_y * uv_stride + uv_x * 2 + 1] = V
  const int uv_offset = uv_y * uv_stride + uv_x * 2;
  const float U =
      static_cast<float>(uv_plane[uv_offset + (swap_uv ? 1 : 0)]) - 128.0f;
  const float V =
      static_cast<float>(uv_plane[uv_offset + (swap_uv ? 0 : 1)]) - 128.0f;

  // YUV (BT.601) 到 BGR 转换
  // R = Y + 1.402 * V
  // G = Y - 0.344 * U - 0.714 * V
  // B = Y + 1.772 * U
  const float R = Y + 1.402f * V;
  const float G = Y - 0.344f * U - 0.714f * V;
  const float B = Y + 1.772f * U;

  // 写入 BGR（注意：BGR 顺序是 B, G, R）
  const int bgr_offset = y * bgr_stride + x * 3;
  bgr[bgr_offset + 0] = clamp_to_uint8(B);
  bgr[bgr_offset + 1] = clamp_to_uint8(G);
  bgr[bgr_offset + 2] = clamp_to_uint8(R);
}

// CUDA kernel: BGR 转换为 GXF NV12
__global__ void BGRToGxfNV12_YPlaneKernel(uint8_t *__restrict__ y_plane,
                                          const uint8_t *__restrict__ bgr,
                                          int width, int height, int y_stride,
                                          int bgr_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const int bgr_offset = y * bgr_stride + x * 3;
  const float B = static_cast<float>(bgr[bgr_offset + 0]);
  const float G = static_cast<float>(bgr[bgr_offset + 1]);
  const float R = static_cast<float>(bgr[bgr_offset + 2]);

  // BGR 到 Y (BT.601)
  // Y = 0.299 * R + 0.587 * G + 0.114 * B
  const float Y = 0.299f * R + 0.587f * G + 0.114f * B;
  y_plane[y * y_stride + x] = clamp_to_uint8(Y);
}

// CUDA kernel: BGR 转换为 GXF NV12 UV 平面（每 2x2 像素块一个 UV 对）
__global__ void BGRToGxfNV12_UVPlaneKernel(uint8_t *__restrict__ uv_plane,
                                           const uint8_t *__restrict__ bgr,
                                           int width, int height, int uv_stride,
                                           int bgr_stride) {
  const int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int uv_y = blockIdx.y * blockDim.y + threadIdx.y;

  const int uv_width = width / 2;
  const int uv_height = height / 2;

  if (uv_x >= uv_width || uv_y >= uv_height)
    return;

  // 对应的 2x2 像素块左上角坐标
  const int y_base = uv_y * 2;
  const int x_base = uv_x * 2;

  // 计算 2x2 块的平均 BGR 值
  float B_sum = 0.0f, G_sum = 0.0f, R_sum = 0.0f;
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int px = x_base + dx;
      const int py = y_base + dy;
      if (px < width && py < height) {
        const int bgr_offset = py * bgr_stride + px * 3;
        B_sum += static_cast<float>(bgr[bgr_offset + 0]);
        G_sum += static_cast<float>(bgr[bgr_offset + 1]);
        R_sum += static_cast<float>(bgr[bgr_offset + 2]);
      }
    }
  }
  const float B = B_sum / 4.0f;
  const float G = G_sum / 4.0f;
  const float R = R_sum / 4.0f;

  // BGR 到 UV (BT.601)
  // U = -0.169 * R - 0.331 * G + 0.500 * B + 128
  // V =  0.500 * R - 0.419 * G - 0.081 * B + 128
  const float U = -0.169f * R - 0.331f * G + 0.500f * B + 128.0f;
  const float V = 0.500f * R - 0.419f * G - 0.081f * B + 128.0f;

  // 写入 UV（GXF 格式：UV 交错）
  const int uv_offset = uv_y * uv_stride + uv_x * 2;
  uv_plane[uv_offset + 0] = clamp_to_uint8(U);
  uv_plane[uv_offset + 1] = clamp_to_uint8(V);
}

// ============== 旧的 UV 平面压缩/扩展函数（保留兼容性）==============

// CUDA kernel: 压缩 GXF NV12 UV 平面为标准 NV12 格式（去掉 padding）
__global__ void CompressUVPlaneKernel(uint8_t *__restrict__ dst,
                                      const uint8_t *__restrict__ src,
                                      int width, int uv_height, int src_stride,
                                      int dst_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < uv_height) {
    dst[y * dst_stride + x] = src[y * src_stride + x];
  }
}

// CUDA kernel: 扩展标准 NV12 UV 平面为 GXF 格式（添加 padding）
__global__ void ExpandUVPlaneKernel(uint8_t *__restrict__ dst,
                                    const uint8_t *__restrict__ src, int width,
                                    int uv_height, int src_stride,
                                    int dst_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y < uv_height) {
    if (x < width) {
      dst[y * dst_stride + x] = src[y * src_stride + x];
    } else if (x < dst_stride) {
      dst[y * dst_stride + x] = 0;
    }
  }
}

extern "C" {

// ============== 新的直接转换函数 ==============

// GXF NV12/NV21 直接转换为 BGR（GPU 到 GPU）
cudaError_t ConvertGxfNV12ToBGREx(void *bgr_gpu, const void *nv12_gpu,
                                  int width, int height, int y_stride,
                                  int uv_stride, int bgr_stride, int swap_uv,
                                  cudaStream_t stream) {
  // GXF NV12 布局：Y 平面 + UV 平面
  const size_t y_size = static_cast<size_t>(y_stride) * height;
  const uint8_t *y_plane = static_cast<const uint8_t *>(nv12_gpu);
  const uint8_t *uv_plane = y_plane + y_size;

  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GxfNV12ToBGRKernel<<<grid, block, 0, stream>>>(
      static_cast<uint8_t *>(bgr_gpu), y_plane, uv_plane, width, height,
      y_stride, uv_stride, bgr_stride, swap_uv);

  return cudaGetLastError();
}

// GXF NV12 直接转换为 BGR（GPU 到 GPU）
cudaError_t ConvertGxfNV12ToBGR(void *bgr_gpu, const void *nv12_gpu, int width,
                                int height, int y_stride, int uv_stride,
                                int bgr_stride, cudaStream_t stream) {
  return ConvertGxfNV12ToBGREx(bgr_gpu, nv12_gpu, width, height, y_stride,
                               uv_stride, bgr_stride, 0, stream);
}

// BGR 转换为 GXF NV12（GPU 到 GPU）
cudaError_t ConvertBGRToGxfNV12(void *nv12_gpu, const void *bgr_gpu, int width,
                                int height, int y_stride, int uv_stride,
                                int bgr_stride, cudaStream_t stream) {
  // GXF NV12 布局
  const size_t y_size = static_cast<size_t>(y_stride) * height;
  uint8_t *y_plane = static_cast<uint8_t *>(nv12_gpu);
  uint8_t *uv_plane = y_plane + y_size;

  dim3 block(32, 8);

  // 转换 Y 平面
  {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    BGRToGxfNV12_YPlaneKernel<<<grid, block, 0, stream>>>(
        y_plane, static_cast<const uint8_t *>(bgr_gpu), width, height, y_stride,
        bgr_stride);
  }

  // 转换 UV 平面
  {
    const int uv_width = width / 2;
    const int uv_height = height / 2;
    dim3 grid((uv_width + block.x - 1) / block.x,
              (uv_height + block.y - 1) / block.y);
    BGRToGxfNV12_UVPlaneKernel<<<grid, block, 0, stream>>>(
        uv_plane, static_cast<const uint8_t *>(bgr_gpu), width, height,
        uv_stride, bgr_stride);
  }

  return cudaGetLastError();
}

// ============== 旧函数（保留兼容性）==============

cudaError_t CompressGxfUVPlane(void *dst, const void *src, int width,
                               int height, int src_stride, int dst_stride,
                               cudaStream_t stream) {
  const int uv_height = height / 2;
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x,
            (uv_height + block.y - 1) / block.y);

  CompressUVPlaneKernel<<<grid, block, 0, stream>>>(
      static_cast<uint8_t *>(dst), static_cast<const uint8_t *>(src), width,
      uv_height, src_stride, dst_stride);

  return cudaGetLastError();
}

cudaError_t ExpandToGxfUVPlane(void *dst, const void *src, int width,
                               int height, int src_stride, int dst_stride,
                               cudaStream_t stream) {
  const int uv_height = height / 2;
  dim3 block(32, 8);
  dim3 grid((dst_stride + block.x - 1) / block.x,
            (uv_height + block.y - 1) / block.y);

  ExpandUVPlaneKernel<<<grid, block, 0, stream>>>(
      static_cast<uint8_t *>(dst), static_cast<const uint8_t *>(src), width,
      uv_height, src_stride, dst_stride);

  return cudaGetLastError();
}

} // extern "C"
