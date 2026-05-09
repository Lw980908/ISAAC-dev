// GXF NV12 格式转换 CUDA 函数声明
// 用于 GXF NV12 格式与 BGR 之间的直接转换（绕过 VPI）

#ifndef GXF_NV12_CONVERT_H
#define GXF_NV12_CONVERT_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============== GXF NV12 <-> BGR 直接转换（推荐使用）==============

/**
 * @brief GXF NV12 直接转换为 BGR（GPU 到 GPU）
 *
 * 使用 CUDA kernel 直接在 GPU 上完成 GXF NV12 到 BGR 的颜色空间转换，
 * 无需中间转换步骤，性能最优。
 *
 * GXF NV12 格式：
 * - Y 平面：stride = width
 * - UV 平面：stride = width * 2（交错存储，后 width 字节是 padding）
 *
 * @param bgr_gpu     输出：BGR GPU 缓冲区（width * height * 3 字节）
 * @param nv12_gpu    输入：GXF NV12 GPU 缓冲区
 * @param width       图像宽度
 * @param height      图像高度
 * @param y_stride    Y 平面 stride（通常 = width）
 * @param uv_stride   UV 平面 stride（GXF = width * 2）
 * @param bgr_stride  BGR 输出 stride（通常 = width * 3）
 * @param stream      CUDA stream（nullptr = 默认 stream）
 * @return cudaError_t  cudaSuccess 表示成功
 */
cudaError_t ConvertGxfNV12ToBGR(void *bgr_gpu, const void *nv12_gpu, int width,
                                int height, int y_stride, int uv_stride,
                                int bgr_stride, cudaStream_t stream);

/**
 * @brief BGR 转换为 GXF NV12（GPU 到 GPU）
 *
 * @param nv12_gpu    输出：GXF NV12 GPU 缓冲区
 * @param bgr_gpu     输入：BGR GPU 缓冲区
 * @param width       图像宽度
 * @param height      图像高度
 * @param y_stride    Y 平面 stride（通常 = width）
 * @param uv_stride   UV 平面 stride（GXF = width * 2）
 * @param bgr_stride  BGR 输入 stride（通常 = width * 3）
 * @param stream      CUDA stream（nullptr = 默认 stream）
 * @return cudaError_t  cudaSuccess 表示成功
 */
cudaError_t ConvertBGRToGxfNV12(void *nv12_gpu, const void *bgr_gpu, int width,
                                int height, int y_stride, int uv_stride,
                                int bgr_stride, cudaStream_t stream);

// ============== 旧函数（保留兼容性）==============

/**
 * @brief 将 GXF NV12 UV 平面压缩为标准 NV12 格式
 */
cudaError_t CompressGxfUVPlane(void *dst, const void *src, int width,
                               int height, int src_stride, int dst_stride,
                               cudaStream_t stream);

/**
 * @brief 将标准 NV12 UV 平面扩展为 GXF 格式
 */
cudaError_t ExpandToGxfUVPlane(void *dst, const void *src, int width,
                               int height, int src_stride, int dst_stride,
                               cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GXF_NV12_CONVERT_H
