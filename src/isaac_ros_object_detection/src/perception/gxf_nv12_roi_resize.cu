#include "gxf_nv12_roi_resize.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint8_t clamp_to_u8(float v) {
  return static_cast<uint8_t>(fminf(fmaxf(v, 0.0f), 255.0f));
}

__device__ __forceinline__ void load_uv_gxf(const uint8_t *uv_plane,
                                            int uv_stride, int uv_x, int uv_y,
                                            float &u, float &v) {
  const int off = uv_y * uv_stride + uv_x * 2;
  u = static_cast<float>(uv_plane[off + 0]);
  v = static_cast<float>(uv_plane[off + 1]);
}

__global__ void
GxfNV12RoiResizeToBGRKernel(uint8_t *__restrict__ dst_bgr, int dst_w, int dst_h,
                            int dst_stride, const uint8_t *__restrict__ y_plane,
                            const uint8_t *__restrict__ uv_plane, int src_w,
                            int src_h, int y_stride, int uv_stride, int roi_x,
                            int roi_y, int roi_w, int roi_h) {
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h) {
    return;
  }

  const float scale_x = static_cast<float>(roi_w) / static_cast<float>(dst_w);
  const float scale_y = static_cast<float>(roi_h) / static_cast<float>(dst_h);
  float sx = roi_x + (dx + 0.5f) * scale_x - 0.5f;
  float sy = roi_y + (dy + 0.5f) * scale_y - 0.5f;
  sx = fminf(fmaxf(sx, 0.0f), static_cast<float>(src_w - 1));
  sy = fminf(fmaxf(sy, 0.0f), static_cast<float>(src_h - 1));

  const int x0 = static_cast<int>(floorf(sx));
  const int y0 = static_cast<int>(floorf(sy));
  const int x1 = min(x0 + 1, src_w - 1);
  const int y1 = min(y0 + 1, src_h - 1);
  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);

  const float y00 = static_cast<float>(y_plane[y0 * y_stride + x0]);
  const float y01 = static_cast<float>(y_plane[y0 * y_stride + x1]);
  const float y10 = static_cast<float>(y_plane[y1 * y_stride + x0]);
  const float y11 = static_cast<float>(y_plane[y1 * y_stride + x1]);
  const float y0v = y00 + (y01 - y00) * fx;
  const float y1v = y10 + (y11 - y10) * fx;
  const float Y = y0v + (y1v - y0v) * fy;

  const int uv_w = src_w / 2;
  const int uv_h = src_h / 2;
  float su = sx * 0.5f;
  float sv = sy * 0.5f;
  su = fminf(fmaxf(su, 0.0f), static_cast<float>(uv_w - 1));
  sv = fminf(fmaxf(sv, 0.0f), static_cast<float>(uv_h - 1));
  const int ux0 = static_cast<int>(floorf(su));
  const int uy0 = static_cast<int>(floorf(sv));
  const int ux1 = min(ux0 + 1, uv_w - 1);
  const int uy1 = min(uy0 + 1, uv_h - 1);
  const float ufx = su - static_cast<float>(ux0);
  const float ufy = sv - static_cast<float>(uy0);

  float u00, v00, u01, v01, u10, v10, u11, v11;
  load_uv_gxf(uv_plane, uv_stride, ux0, uy0, u00, v00);
  load_uv_gxf(uv_plane, uv_stride, ux1, uy0, u01, v01);
  load_uv_gxf(uv_plane, uv_stride, ux0, uy1, u10, v10);
  load_uv_gxf(uv_plane, uv_stride, ux1, uy1, u11, v11);

  const float u0v = u00 + (u01 - u00) * ufx;
  const float u1v = u10 + (u11 - u10) * ufx;
  const float v0v = v00 + (v01 - v00) * ufx;
  const float v1v = v10 + (v11 - v10) * ufx;
  const float U = (u0v + (u1v - u0v) * ufy) - 128.0f;
  const float V = (v0v + (v1v - v0v) * ufy) - 128.0f;

  const float R = Y + 1.402f * V;
  const float G = Y - 0.344f * U - 0.714f * V;
  const float B = Y + 1.772f * U;

  const int dst_off = dy * dst_stride + dx * 3;
  dst_bgr[dst_off + 0] = clamp_to_u8(B);
  dst_bgr[dst_off + 1] = clamp_to_u8(G);
  dst_bgr[dst_off + 2] = clamp_to_u8(R);
}

extern "C" cudaError_t
ConvertGxfNV12RoiResizeToBGR(void *dst_bgr, int dst_width, int dst_height,
                             int dst_bgr_stride, const void *src_nv12,
                             int src_width, int src_height, int src_y_stride,
                             int src_uv_stride, int roi_x, int roi_y, int roi_w,
                             int roi_h, cudaStream_t stream) {
  if (!dst_bgr || !src_nv12 || dst_width <= 0 || dst_height <= 0 ||
      src_width <= 0 || src_height <= 0) {
    return cudaSuccess;
  }

  if (roi_w <= 0 || roi_h <= 0) {
    return cudaSuccess;
  }

  roi_x = std::max(0, std::min(roi_x, src_width - 1));
  roi_y = std::max(0, std::min(roi_y, src_height - 1));
  roi_w = std::max(1, std::min(roi_w, src_width - roi_x));
  roi_h = std::max(1, std::min(roi_h, src_height - roi_y));

  const size_t y_size = static_cast<size_t>(src_y_stride) * src_height;
  const auto *y_plane = static_cast<const uint8_t *>(src_nv12);
  const auto *uv_plane = y_plane + y_size;

  dim3 block(32, 8);
  dim3 grid((dst_width + block.x - 1) / block.x,
            (dst_height + block.y - 1) / block.y);
  GxfNV12RoiResizeToBGRKernel<<<grid, block, 0, stream>>>(
      static_cast<uint8_t *>(dst_bgr), dst_width, dst_height, dst_bgr_stride,
      y_plane, uv_plane, src_width, src_height, src_y_stride, src_uv_stride,
      roi_x, roi_y, roi_w, roi_h);
  return cudaGetLastError();
}
