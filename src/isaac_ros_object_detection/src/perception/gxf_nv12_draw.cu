// 在 GXF NV12 图像上绘制矩形框（CUDA kernel）

#include "gxf_nv12_draw.h"

#include <cuda_runtime.h>

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ===== 简易 5x7 字模（bit=1 表示填充像素）=====
// 仅实现：'F','P','S',':',' ','0'~'9','.'
__device__ __forceinline__ uint8_t font5x7_row(char c, int row) {
  // 每行 5bit（从高位到低位依次为列 0..4）
  switch (c) {
  case 'D': {
    static const uint8_t r[7] = {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    return r[row];
  }
  case 'H': {
    static const uint8_t r[7] = {0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x11};
    return r[row];
  }
  case '=': {
    static const uint8_t r[7] = {0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00};
    return r[row];
  }
  case 'F': {
    // #####
    // #
    // ####
    // #
    // #
    // #
    // #
    static const uint8_t r[7] = {0x1F, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x10};
    return r[row];
  }
  case 'P': {
    // ####
    // #  #
    // ####
    // #
    // #
    // #
    // #
    static const uint8_t r[7] = {0x1E, 0x11, 0x1E, 0x10, 0x10, 0x10, 0x10};
    return r[row];
  }
  case 'S': {
    // #####
    // #
    // #####
    //     #
    // #####
    static const uint8_t r[7] = {0x1F, 0x10, 0x1F, 0x01, 0x1F, 0x00, 0x00};
    return r[row];
  }
  case ':': {
    static const uint8_t r[7] = {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00};
    return r[row];
  }
  case '.': {
    static const uint8_t r[7] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00};
    return r[row];
  }
  case '-': {
    static const uint8_t r[7] = {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    return r[row];
  }
  case 'm': {
    static const uint8_t r[7] = {0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x00};
    return r[row];
  }
  case ' ': {
    return 0x00;
  }
  default:
    break;
  }

  if (c >= '0' && c <= '9') {
    switch (c) {
    case '0': {
      static const uint8_t r[7] = {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
      return r[row];
    }
    case '1': {
      static const uint8_t r[7] = {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
      return r[row];
    }
    case '2': {
      static const uint8_t r[7] = {0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F};
      return r[row];
    }
    case '3': {
      static const uint8_t r[7] = {0x1F, 0x01, 0x06, 0x01, 0x01, 0x11, 0x0E};
      return r[row];
    }
    case '4': {
      static const uint8_t r[7] = {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
      return r[row];
    }
    case '5': {
      static const uint8_t r[7] = {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E};
      return r[row];
    }
    case '6': {
      static const uint8_t r[7] = {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E};
      return r[row];
    }
    case '7': {
      static const uint8_t r[7] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
      return r[row];
    }
    case '8': {
      static const uint8_t r[7] = {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
      return r[row];
    }
    case '9': {
      static const uint8_t r[7] = {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C};
      return r[row];
    }
    default:
      break;
    }
  }

  return 0x00;
}

__device__ __forceinline__ void
nv12_set_pixel(uint8_t *__restrict__ y_plane, uint8_t *__restrict__ uv_plane,
               int width, int height, int y_stride, int uv_stride, int x, int y,
               uint8_t y_val, uint8_t u_val, uint8_t v_val) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  y_plane[y * y_stride + x] = y_val;
  const int uv_w = (width + 1) >> 1;
  const int uv_h = (height + 1) >> 1;
  const int ux = x >> 1;
  const int uy = y >> 1;
  if (ux >= 0 && ux < uv_w && uy >= 0 && uy < uv_h) {
    const int off = uy * uv_stride + ux * 2;
    uv_plane[off + 0] = u_val;
    uv_plane[off + 1] = v_val;
  }
}

__global__ void DrawGxfNV12FpsKernel(uint8_t *__restrict__ y_plane,
                                     uint8_t *__restrict__ uv_plane, int width,
                                     int height, int y_stride, int uv_stride,
                                     int fps_x10, int start_x, int start_y,
                                     int scale, uint8_t y_val, uint8_t u_val,
                                     uint8_t v_val) {
  // 文本固定： "FPS:" + 空格 + 3位整数(可空格) + '.' + 1位小数  => 10字符
  constexpr int kCharW = 5;
  constexpr int kCharH = 7;
  constexpr int kGap = 1;
  constexpr int kLen = 10;
  const int pitch = (kCharW + kGap) * scale;
  const int total_w = kLen * pitch;
  const int total_h = kCharH * scale;

  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= total_w || py >= total_h) {
    return;
  }

  const int x = start_x + px;
  const int y = start_y + py;
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }

  const int char_idx = px / pitch;
  const int in_char_x = (px % pitch) / scale;
  const int in_char_y = py / scale;
  if (char_idx < 0 || char_idx >= kLen) {
    return;
  }
  if (in_char_x >= kCharW || in_char_y >= kCharH) {
    return;
  }

  // 组装字符序列
  const int fps_int = fps_x10 / 10;
  const int fps_frac = fps_x10 - fps_int * 10;
  const int hundreds = fps_int / 100;
  const int tens = (fps_int / 10) % 10;
  const int ones = fps_int % 10;

  char c = ' ';
  switch (char_idx) {
  case 0:
    c = 'F';
    break;
  case 1:
    c = 'P';
    break;
  case 2:
    c = 'S';
    break;
  case 3:
    c = ':';
    break;
  case 4:
    c = ' ';
    break;
  case 5:
    c = (hundreds > 0) ? static_cast<char>('0' + hundreds) : ' ';
    break;
  case 6:
    c = (hundreds > 0 || tens > 0) ? static_cast<char>('0' + tens) : ' ';
    break;
  case 7:
    c = static_cast<char>('0' + ones);
    break;
  case 8:
    c = '.';
    break;
  case 9:
    c = static_cast<char>('0' + fps_frac);
    break;
  default:
    c = ' ';
    break;
  }

  const uint8_t mask = font5x7_row(c, in_char_y);
  const int bit = (mask >> (kCharW - 1 - in_char_x)) & 0x01;
  if (!bit) {
    return;
  }

  y_plane[y * y_stride + x] = y_val;
  // 性能优化：FPS 文字只改亮度(Y)，不改色度(UV)
}

// 对应一个 bbox 的绘制：每个 block 处理一个 bbox
__global__ void DrawGxfNV12BboxesKernel(
    uint8_t *__restrict__ y_plane, uint8_t *__restrict__ uv_plane, int width,
    int height, int y_stride, int uv_stride, const GxfBBox *__restrict__ boxes,
    int num_boxes, int thickness, uint8_t y_val, uint8_t u_val, uint8_t v_val) {
  const int bi = blockIdx.x;
  if (bi >= num_boxes) {
    return;
  }

  int l = boxes[bi].left;
  int t = boxes[bi].top;
  int r = boxes[bi].right;
  int b = boxes[bi].bottom;

  // 规范化 & clamp
  if (l > r) {
    int tmp = l;
    l = r;
    r = tmp;
  }
  if (t > b) {
    int tmp = t;
    t = b;
    b = tmp;
  }
  l = clamp_int(l, 0, width - 1);
  r = clamp_int(r, 0, width - 1);
  t = clamp_int(t, 0, height - 1);
  b = clamp_int(b, 0, height - 1);

  if (r - l < 2 || b - t < 2) {
    return;
  }

  thickness = clamp_int(thickness, 1, 8);

  const int tid = threadIdx.x;
  const int step = blockDim.x;
  const int uv_w = (width + 1) >> 1;
  const int uv_h = (height + 1) >> 1;

  for (int x = l + tid; x <= r; x += step) {
    for (int k = 0; k < thickness; ++k) {
      const int y_top = t + k;
      const int y_bot = b - k;

      if (y_top >= 0 && y_top < height) {
        y_plane[y_top * y_stride + x] = y_val;
        const int ux = x >> 1;
        const int uy = y_top >> 1;
        if (ux >= 0 && ux < uv_w && uy >= 0 && uy < uv_h) {
          const int off = uy * uv_stride + ux * 2;
          uv_plane[off + 0] = u_val;
          uv_plane[off + 1] = v_val;
        }
      }

      if (y_bot >= 0 && y_bot < height && y_bot != y_top) {
        y_plane[y_bot * y_stride + x] = y_val;
        const int ux = x >> 1;
        const int uy = y_bot >> 1;
        if (ux >= 0 && ux < uv_w && uy >= 0 && uy < uv_h) {
          const int off = uy * uv_stride + ux * 2;
          uv_plane[off + 0] = u_val;
          uv_plane[off + 1] = v_val;
        }
      }
    }
  }

  for (int y = t + tid; y <= b; y += step) {
    for (int k = 0; k < thickness; ++k) {
      const int x_left = l + k;
      const int x_right = r - k;

      if (x_left >= 0 && x_left < width) {
        y_plane[y * y_stride + x_left] = y_val;
        const int ux = x_left >> 1;
        const int uy = y >> 1;
        if (ux >= 0 && ux < uv_w && uy >= 0 && uy < uv_h) {
          const int off = uy * uv_stride + ux * 2;
          uv_plane[off + 0] = u_val;
          uv_plane[off + 1] = v_val;
        }
      }

      if (x_right >= 0 && x_right < width && x_right != x_left) {
        y_plane[y * y_stride + x_right] = y_val;
        const int ux = x_right >> 1;
        const int uy = y >> 1;
        if (ux >= 0 && ux < uv_w && uy >= 0 && uy < uv_h) {
          const int off = uy * uv_stride + ux * 2;
          uv_plane[off + 0] = u_val;
          uv_plane[off + 1] = v_val;
        }
      }
    }
  }
}

__global__ void DrawGxfNV12DashedLinesKernel(
    uint8_t *__restrict__ y_plane, uint8_t *__restrict__ uv_plane, int width,
    int height, int y_stride, int uv_stride,
    const GxfDashedLine *__restrict__ lines, int num_lines) {
  const int li = blockIdx.x;
  if (li >= num_lines) {
    return;
  }
  const GxfDashedLine ln = lines[li];
  const int dash_len = ln.dash_len > 0 ? ln.dash_len : 1;
  const int gap_len = ln.gap_len >= 0 ? ln.gap_len : 0;
  const int pattern = dash_len + gap_len;
  const int thickness = clamp_int(ln.thickness, 1, 8);

  if (ln.y0 == ln.y1) {
    int x0 = ln.x0;
    int x1 = ln.x1;
    if (x0 > x1) {
      int tmp = x0;
      x0 = x1;
      x1 = tmp;
    }
    const int len = x1 - x0 + 1;
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    for (int i = tid; i < len; i += step) {
      if (pattern > 0 && (i % pattern) >= dash_len) {
        continue;
      }
      const int x = x0 + i;
      for (int k = 0; k < thickness; ++k) {
        const int y = ln.y0 - (thickness / 2) + k;
        nv12_set_pixel(y_plane, uv_plane, width, height, y_stride, uv_stride, x,
                       y, ln.y_val, ln.u_val, ln.v_val);
      }
    }
    return;
  }

  if (ln.x0 == ln.x1) {
    int y0 = ln.y0;
    int y1 = ln.y1;
    if (y0 > y1) {
      int tmp = y0;
      y0 = y1;
      y1 = tmp;
    }
    const int len = y1 - y0 + 1;
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    for (int i = tid; i < len; i += step) {
      if (pattern > 0 && (i % pattern) >= dash_len) {
        continue;
      }
      const int y = y0 + i;
      for (int k = 0; k < thickness; ++k) {
        const int x = ln.x0 - (thickness / 2) + k;
        nv12_set_pixel(y_plane, uv_plane, width, height, y_stride, uv_stride, x,
                       y, ln.y_val, ln.u_val, ln.v_val);
      }
    }
    return;
  }

  // General line (DDA). Supports slanted edges for 3D wireframe overlays.
  const int dx = ln.x1 - ln.x0;
  const int dy = ln.y1 - ln.y0;
  const int adx = dx >= 0 ? dx : -dx;
  const int ady = dy >= 0 ? dy : -dy;
  const int n = adx > ady ? adx : ady; // number of steps
  if (n <= 0) {
    nv12_set_pixel(y_plane, uv_plane, width, height, y_stride, uv_stride, ln.x0,
                   ln.y0, ln.y_val, ln.u_val, ln.v_val);
    return;
  }

  const int tid = threadIdx.x;
  const int step = blockDim.x;
  const int half = thickness / 2;
  for (int i = tid; i <= n; i += step) {
    if (pattern > 0 && (i % pattern) >= dash_len) {
      continue;
    }
    const float t = static_cast<float>(i) / static_cast<float>(n);
    const int x = __float2int_rn(static_cast<float>(ln.x0) +
                                 static_cast<float>(dx) * t);
    const int y = __float2int_rn(static_cast<float>(ln.y0) +
                                 static_cast<float>(dy) * t);
    for (int oy = -half; oy <= half; ++oy) {
      for (int ox = -half; ox <= half; ++ox) {
        nv12_set_pixel(y_plane, uv_plane, width, height, y_stride, uv_stride,
                       x + ox, y + oy, ln.y_val, ln.u_val, ln.v_val);
      }
    }
  }
}

__global__ void DrawGxfNV12TextMarksKernel(
    uint8_t *__restrict__ y_plane, uint8_t *__restrict__ uv_plane, int width,
    int height, int y_stride, int uv_stride,
    const GxfTextMark *__restrict__ marks, int num_marks) {
  const int mi = blockIdx.x;
  if (mi >= num_marks) {
    return;
  }
  const GxfTextMark mk = marks[mi];
  const int scale = clamp_int(mk.scale, 1, 6);
  constexpr int kCharW = 5;
  constexpr int kCharH = 7;
  constexpr int kGap = 1;
  const int pitch = (kCharW + kGap) * scale;

  int len = 0;
  for (int i = 0; i < 15; ++i) {
    if (mk.text[i] == '\0') {
      break;
    }
    len++;
  }
  if (len <= 0) {
    return;
  }

  const int total_w = len * pitch;
  const int total_h = kCharH * scale;

  const int px = blockIdx.y * blockDim.x + threadIdx.x;
  const int py = blockIdx.z * blockDim.y + threadIdx.y;
  if (px >= total_w || py >= total_h) {
    return;
  }

  const int x = mk.x + px;
  const int y = mk.y + py;
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }

  const int char_idx = px / pitch;
  const int in_char_x = (px % pitch) / scale;
  const int in_char_y = py / scale;
  if (char_idx < 0 || char_idx >= len) {
    return;
  }
  if (in_char_x >= kCharW || in_char_y >= kCharH) {
    return;
  }

  const char c = mk.text[char_idx];
  const uint8_t mask = font5x7_row(c, in_char_y);
  const int bit = (mask >> (kCharW - 1 - in_char_x)) & 0x01;
  if (!bit) {
    return;
  }

  nv12_set_pixel(y_plane, uv_plane, width, height, y_stride, uv_stride, x, y,
                 mk.y_val, mk.u_val, mk.v_val);
}

__global__ void DrawGxfNV12CircleMarksKernel(
    uint8_t *__restrict__ y_plane, uint8_t *__restrict__ uv_plane, int width,
    int height, int y_stride, int uv_stride,
    const GxfCircleMark *__restrict__ marks, int num_marks) {
  const int mi = blockIdx.x;
  if (mi >= num_marks) {
    return;
  }

  const int cx = marks[mi].x;
  const int cy = marks[mi].y;
  int r = marks[mi].radius;
  if (r <= 0) {
    return;
  }
  r = clamp_int(r, 1, 32);

  const int tid = threadIdx.x;
  const int step = blockDim.x;

  const int d = r * 2 + 1;
  const int total = d * d;
  const int rr = r * r;
  const int uv_w = (width + 1) >> 1;
  const int uv_h = (height + 1) >> 1;

  for (int idx = tid; idx < total; idx += step) {
    const int dx = (idx % d) - r;
    const int dy = (idx / d) - r;
    if (dx * dx + dy * dy > rr) {
      continue;
    }

    const int x = cx + dx;
    const int y = cy + dy;
    if (x < 0 || x >= width || y < 0 || y >= height) {
      continue;
    }

    y_plane[y * y_stride + x] = marks[mi].y_val;

    const int ux = x >> 1;
    const int uy = y >> 1;
    if (ux < 0 || ux >= uv_w || uy < 0 || uy >= uv_h) {
      continue;
    }
    const int off = uy * uv_stride + ux * 2;
    uv_plane[off + 0] = marks[mi].u_val;
    uv_plane[off + 1] = marks[mi].v_val;
  }
}

extern "C" {

cudaError_t DrawGxfNV12Bboxes(void *nv12_gpu, int width, int height,
                              int y_stride, int uv_stride,
                              const GxfBBox *bboxes_gpu, int num_bboxes,
                              int thickness, uint8_t y_val, uint8_t u_val,
                              uint8_t v_val, cudaStream_t stream) {
  if (!nv12_gpu || !bboxes_gpu || num_bboxes <= 0 || width <= 0 ||
      height <= 0) {
    return cudaSuccess;
  }

  const size_t y_size = static_cast<size_t>(y_stride) * height;
  auto *y_plane = static_cast<uint8_t *>(nv12_gpu);
  auto *uv_plane = y_plane + y_size;

  const int threads = 256;
  DrawGxfNV12BboxesKernel<<<num_bboxes, threads, 0, stream>>>(
      y_plane, uv_plane, width, height, y_stride, uv_stride, bboxes_gpu,
      num_bboxes, thickness, y_val, u_val, v_val);
  return cudaGetLastError();
}

cudaError_t DrawGxfNV12Fps(void *nv12_gpu, int width, int height, int y_stride,
                           int uv_stride, int fps_x10, cudaStream_t stream) {
  if (!nv12_gpu || width <= 0 || height <= 0) {
    return cudaSuccess;
  }
  const size_t y_size = static_cast<size_t>(y_stride) * height;
  auto *y_plane = static_cast<uint8_t *>(nv12_gpu);
  auto *uv_plane = y_plane + y_size;

  // 右上角，留边距
  constexpr int margin = 10;
  constexpr int scale = 2; // 放大让文字可读
  constexpr int kCharW = 5;
  constexpr int kCharH = 7;
  constexpr int kGap = 1;
  constexpr int kLen = 10;
  const int pitch = (kCharW + kGap) * scale;
  const int total_w = kLen * pitch;
  const int total_h = kCharH * scale;
  const int start_x = width - margin - total_w;
  const int start_y = margin;

  // 颜色：白色偏亮（Y高）+ UV 置中（灰/白）
  constexpr uint8_t y_val = 235;
  constexpr uint8_t u_val = 128;
  constexpr uint8_t v_val = 128;

  dim3 block(16, 16);
  dim3 grid((total_w + block.x - 1) / block.x,
            (total_h + block.y - 1) / block.y);
  DrawGxfNV12FpsKernel<<<grid, block, 0, stream>>>(
      y_plane, uv_plane, width, height, y_stride, uv_stride, fps_x10, start_x,
      start_y, scale, y_val, u_val, v_val);
  return cudaGetLastError();
}

cudaError_t DrawGxfNV12CircleMarks(void *nv12_gpu, int width, int height,
                                   int y_stride, int uv_stride,
                                   const GxfCircleMark *marks_gpu,
                                   int num_marks, cudaStream_t stream) {
  if (!nv12_gpu || !marks_gpu || num_marks <= 0 || width <= 0 || height <= 0) {
    return cudaSuccess;
  }
  const size_t y_size = static_cast<size_t>(y_stride) * height;
  auto *y_plane = static_cast<uint8_t *>(nv12_gpu);
  auto *uv_plane = y_plane + y_size;

  const int threads = 256;
  DrawGxfNV12CircleMarksKernel<<<num_marks, threads, 0, stream>>>(
      y_plane, uv_plane, width, height, y_stride, uv_stride, marks_gpu,
      num_marks);
  return cudaGetLastError();
}

cudaError_t DrawGxfNV12DashedLines(void *nv12_gpu, int width, int height,
                                   int y_stride, int uv_stride,
                                   const GxfDashedLine *lines_gpu,
                                   int num_lines, cudaStream_t stream) {
  if (!nv12_gpu || !lines_gpu || num_lines <= 0 || width <= 0 || height <= 0) {
    return cudaSuccess;
  }
  const size_t y_size = static_cast<size_t>(y_stride) * height;
  auto *y_plane = static_cast<uint8_t *>(nv12_gpu);
  auto *uv_plane = y_plane + y_size;

  const int threads = 256;
  DrawGxfNV12DashedLinesKernel<<<num_lines, threads, 0, stream>>>(
      y_plane, uv_plane, width, height, y_stride, uv_stride, lines_gpu,
      num_lines);
  return cudaGetLastError();
}

cudaError_t DrawGxfNV12TextMarks(void *nv12_gpu, int width, int height,
                                 int y_stride, int uv_stride,
                                 const GxfTextMark *marks_gpu, int num_marks,
                                 cudaStream_t stream) {
  if (!nv12_gpu || !marks_gpu || num_marks <= 0 || width <= 0 || height <= 0) {
    return cudaSuccess;
  }
  const size_t y_size = static_cast<size_t>(y_stride) * height;
  auto *y_plane = static_cast<uint8_t *>(nv12_gpu);
  auto *uv_plane = y_plane + y_size;

  constexpr int kCharW = 5;
  constexpr int kCharH = 7;
  constexpr int kGap = 1;
  const int max_len = 15;
  const int max_scale = 6;
  const int pitch = (kCharW + kGap) * max_scale;
  const int total_w = max_len * pitch;
  const int total_h = kCharH * max_scale;

  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned int>(num_marks),
            static_cast<unsigned int>((total_w + block.x - 1) / block.x),
            static_cast<unsigned int>((total_h + block.y - 1) / block.y));
  DrawGxfNV12TextMarksKernel<<<grid, block, 0, stream>>>(
      y_plane, uv_plane, width, height, y_stride, uv_stride, marks_gpu,
      num_marks);
  return cudaGetLastError();
}

} // extern "C"
