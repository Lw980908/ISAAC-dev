// GXF NV12 上绘制检测框（GPU 侧）
// 目标：NV12 直推理模式下，不回退 CPU，也能叠框并发布 NITROS NV12

#ifndef GXF_NV12_DRAW_H
#define GXF_NV12_DRAW_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 轻量 bbox（避免直接使用含 std::vector 的 Box）
typedef struct GxfBBox {
  int left;
  int top;
  int right;
  int bottom;
} GxfBBox;

typedef struct GxfCircleMark {
  int x;
  int y;
  int radius;
  uint8_t y_val;
  uint8_t u_val;
  uint8_t v_val;
} GxfCircleMark;

typedef struct GxfDashedLine {
  int x0;
  int y0;
  int x1;
  int y1;
  int dash_len;
  int gap_len;
  int thickness;
  uint8_t y_val;
  uint8_t u_val;
  uint8_t v_val;
} GxfDashedLine;

typedef struct GxfTextMark {
  int x;
  int y;
  int scale;
  char text[16];
  uint8_t y_val;
  uint8_t u_val;
  uint8_t v_val;
} GxfTextMark;

/**
 * @brief 在 GXF NV12 图像上绘制若干矩形框（边框）
 *
 * GXF NV12 格式：
 * - Y 平面：stride = width
 * - UV 平面：stride = width * 2（前 width 字节为 NV12 UV 交错，后 width
 * padding）
 */
cudaError_t DrawGxfNV12Bboxes(void *nv12_gpu, int width, int height,
                              int y_stride, int uv_stride,
                              const GxfBBox *bboxes_gpu, int num_bboxes,
                              int thickness, uint8_t y_val, uint8_t u_val,
                              uint8_t v_val, cudaStream_t stream);

/**
 * @brief 在 GXF NV12 图像右上角绘制 FPS 文本（例如 FPS: 29.4）
 *
 * @param fps_x10 FPS * 10（例如 29.4 -> 294）
 */
cudaError_t DrawGxfNV12Fps(void *nv12_gpu, int width, int height, int y_stride,
                           int uv_stride, int fps_x10, cudaStream_t stream);

cudaError_t DrawGxfNV12CircleMarks(void *nv12_gpu, int width, int height,
                                   int y_stride, int uv_stride,
                                   const GxfCircleMark *marks_gpu,
                                   int num_marks, cudaStream_t stream);

cudaError_t DrawGxfNV12DashedLines(void *nv12_gpu, int width, int height,
                                   int y_stride, int uv_stride,
                                   const GxfDashedLine *lines_gpu,
                                   int num_lines, cudaStream_t stream);

cudaError_t DrawGxfNV12TextMarks(void *nv12_gpu, int width, int height,
                                 int y_stride, int uv_stride,
                                 const GxfTextMark *marks_gpu, int num_marks,
                                 cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GXF_NV12_DRAW_H

