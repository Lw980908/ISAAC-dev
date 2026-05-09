// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_v4l2_camera/nv12_format_convert.hpp"

// Y 平面转换 kernel：去除 pitch 对齐
__global__ void ConvertYPlaneKernel(uint8_t *__restrict__ dst,
                                    const uint8_t *__restrict__ src, int width,
                                    int height, int src_pitch, int dst_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    dst[y * dst_stride + x] = src[y * src_pitch + x];
  }
}

// UV 平面转换 kernel：去除 pitch 对齐，并转换为 GXF 期望的 stride
__global__ void ConvertUVPlaneKernel(uint8_t *__restrict__ dst,
                                     const uint8_t *__restrict__ src, int width,
                                     int uv_height, int src_pitch,
                                     int dst_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // UV 平面实际数据宽度 = width（U 和 V 交错，各 width/2）
  if (x < width && y < uv_height) {
    dst[y * dst_stride + x] = src[y * src_pitch + x];
  }
}

extern "C" cudaError_t
ConvertNV12ToGxfLayout(void *dst_buffer, const void *src_y, const void *src_uv,
                       int width, int height, int src_y_pitch, int src_uv_pitch,
                       int dst_y_stride, int dst_uv_stride,
                       cudaStream_t stream) {
  // 计算 Y 平面输出大小（用于定位 UV 平面起始位置）
  const size_t y_size = static_cast<size_t>(dst_y_stride) * height;

  // 使用 32x8 的 block 尺寸（适合图像处理）
  dim3 block(32, 8);

  // 转换 Y 平面
  {
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    ConvertYPlaneKernel<<<grid, block, 0, stream>>>(
        static_cast<uint8_t *>(dst_buffer), static_cast<const uint8_t *>(src_y),
        width, height, src_y_pitch, dst_y_stride);
  }

  // 转换 UV 平面
  {
    const int uv_height = height / 2;
    dim3 grid((width + block.x - 1) / block.x,
              (uv_height + block.y - 1) / block.y);
    ConvertUVPlaneKernel<<<grid, block, 0, stream>>>(
        static_cast<uint8_t *>(dst_buffer) + y_size,
        static_cast<const uint8_t *>(src_uv), width, uv_height, src_uv_pitch,
        dst_uv_stride);
  }

  return cudaGetLastError();
}

// NV12 -> BGR8 kernel（输出紧凑 BGR，stride=width*3）
__device__ __forceinline__ uint8_t ClampToU8(int v) {
  return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__global__ void ConvertNV12ToBgr8Kernel(
    uint8_t *__restrict__ dst_bgr,
    const uint8_t *__restrict__ src_y,
    const uint8_t *__restrict__ src_uv,
    int width, int height,
    int src_y_pitch, int src_uv_pitch,
    int dst_bgr_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  // Luma
  const int Y = static_cast<int>(src_y[y * src_y_pitch + x]);

  // Chroma (NV12: UV interleaved, 2x2 subsampling)
  const int uv_x = (x >> 1) << 1;               // even index (bytes)
  const int uv_y = (y >> 1);
  const int uv_idx = uv_y * src_uv_pitch + uv_x;
  const int U = static_cast<int>(src_uv[uv_idx + 0]);
  const int V = static_cast<int>(src_uv[uv_idx + 1]);

  // BT.601 limited range conversion
  // C = Y - 16, D = U - 128, E = V - 128
  const int C = Y - 16;
  const int D = U - 128;
  const int E = V - 128;

  // Use integer approximation:
  // R = (298*C + 409*E + 128) >> 8
  // G = (298*C - 100*D - 208*E + 128) >> 8
  // B = (298*C + 516*D + 128) >> 8
  const int r = (298 * C + 409 * E + 128) >> 8;
  const int g = (298 * C - 100 * D - 208 * E + 128) >> 8;
  const int b = (298 * C + 516 * D + 128) >> 8;

  const int out_idx = y * dst_bgr_stride + x * 3;
  dst_bgr[out_idx + 0] = ClampToU8(b);
  dst_bgr[out_idx + 1] = ClampToU8(g);
  dst_bgr[out_idx + 2] = ClampToU8(r);
}

extern "C" cudaError_t ConvertNV12ToBgr8(
    void *dst_bgr, const void *src_y, const void *src_uv,
    int width, int height,
    int src_y_pitch, int src_uv_pitch,
    int dst_bgr_stride, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

  ConvertNV12ToBgr8Kernel<<<grid, block, 0, stream>>>(
      static_cast<uint8_t *>(dst_bgr),
      static_cast<const uint8_t *>(src_y),
      static_cast<const uint8_t *>(src_uv),
      width, height, src_y_pitch, src_uv_pitch, dst_bgr_stride);

  return cudaGetLastError();
}

// BGRA8 -> BGR8 kernel（丢弃 A 通道，输出紧凑 BGR，stride=width*3）
__global__ void ConvertBGRA8ToBgr8Kernel(
    uint8_t *__restrict__ dst_bgr,
    const uint8_t *__restrict__ src_bgra,
    int width, int height,
    int src_bgra_pitch,
    int dst_bgr_stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const int in_idx = y * src_bgra_pitch + x * 4;
  const uint8_t b = src_bgra[in_idx + 0];
  const uint8_t g = src_bgra[in_idx + 1];
  const uint8_t r = src_bgra[in_idx + 2];

  const int out_idx = y * dst_bgr_stride + x * 3;
  dst_bgr[out_idx + 0] = b;
  dst_bgr[out_idx + 1] = g;
  dst_bgr[out_idx + 2] = r;
}

extern "C" cudaError_t ConvertBGRA8ToBgr8(
    void *dst_bgr, const void *src_bgra, int width, int height,
    int src_bgra_pitch, int dst_bgr_stride, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

  ConvertBGRA8ToBgr8Kernel<<<grid, block, 0, stream>>>(
      static_cast<uint8_t *>(dst_bgr),
      static_cast<const uint8_t *>(src_bgra),
      width, height, src_bgra_pitch, dst_bgr_stride);

  return cudaGetLastError();
}
