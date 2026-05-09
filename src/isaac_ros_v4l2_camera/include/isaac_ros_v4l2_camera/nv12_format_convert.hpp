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

#ifndef ISAAC_ROS_V4L2_CAMERA__NV12_FORMAT_CONVERT_HPP_
#define ISAAC_ROS_V4L2_CAMERA__NV12_FORMAT_CONVERT_HPP_

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在 GPU 上将 NvBufSurface 的 NV12 布局转换为 GXF 期望的布局
 *
 * 完全在 GPU 上执行，CPU 不接触像素数据。
 *
 * @param dst_buffer    输出：CUDA 设备内存（GXF 布局）
 * @param src_y         输入：Y 平面 GPU 指针（NvBufSurface dataPtr）
 * @param src_uv        输入：UV 平面 GPU 指针
 * @param width         图像宽度
 * @param height        图像高度
 * @param src_y_pitch   源 Y 平面 pitch（字节）
 * @param src_uv_pitch  源 UV 平面 pitch（字节）
 * @param dst_y_stride  目标 Y 平面 stride（= width）
 * @param dst_uv_stride 目标 UV 平面 stride（= width * 2，GXF 要求）
 * @param stream        CUDA stream（nullptr = 默认 stream）
 * @return cudaError_t  cudaSuccess 表示成功
 */
cudaError_t ConvertNV12ToGxfLayout(void *dst_buffer, const void *src_y,
                                   const void *src_uv, int width, int height,
                                   int src_y_pitch, int src_uv_pitch,
                                   int dst_y_stride, int dst_uv_stride,
                                   cudaStream_t stream);

/**
 * @brief 在 GPU 上将 NV12 转换为 BGR8（无 padding，行跨度=width*3）
 *
 * 完全在 GPU 上执行，CPU 不接触像素数据。
 *
 * @param dst_bgr        输出：CUDA 设备内存（BGR8，紧凑排列）
 * @param src_y          输入：Y 平面 GPU 指针
 * @param src_uv         输入：UV 平面 GPU 指针（NV12: U,V 交错）
 * @param width          图像宽度（像素）
 * @param height         图像高度（像素）
 * @param src_y_pitch    源 Y 平面 pitch（字节）
 * @param src_uv_pitch   源 UV 平面 pitch（字节）
 * @param dst_bgr_stride 目标 BGR 平面 stride（= width * 3）
 * @param stream         CUDA stream（nullptr = 默认 stream）
 * @return cudaError_t   cudaSuccess 表示成功
 */
cudaError_t ConvertNV12ToBgr8(void *dst_bgr, const void *src_y,
                              const void *src_uv, int width, int height,
                              int src_y_pitch, int src_uv_pitch,
                              int dst_bgr_stride, cudaStream_t stream);

/**
 * @brief 在 GPU 上将 BGRA8（pitch 可能对齐）转换为 BGR8（无
 * padding，行跨度=width*3）
 *
 * 完全在 GPU 上执行，CPU 不接触像素数据。
 *
 * @param dst_bgr        输出：CUDA 设备内存（BGR8，紧凑排列）
 * @param src_bgra       输入：BGRA 平面 GPU 指针（单平面，4 bytes/pixel）
 * @param width          图像宽度（像素）
 * @param height         图像高度（像素）
 * @param src_bgra_pitch 源 BGRA 平面 pitch（字节）
 * @param dst_bgr_stride 目标 BGR 平面 stride（= width * 3）
 * @param stream         CUDA stream（nullptr = 默认 stream）
 * @return cudaError_t   cudaSuccess 表示成功
 */
cudaError_t ConvertBGRA8ToBgr8(void *dst_bgr, const void *src_bgra, int width,
                               int height, int src_bgra_pitch,
                               int dst_bgr_stride, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // ISAAC_ROS_V4L2_CAMERA__NV12_FORMAT_CONVERT_HPP_
