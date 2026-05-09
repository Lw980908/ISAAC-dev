#ifndef GXF_NV12_ROI_RESIZE_H
#define GXF_NV12_ROI_RESIZE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t ConvertGxfNV12RoiResizeToBGR(void *dst_bgr, int dst_width,
                                         int dst_height, int dst_bgr_stride,
                                         const void *src_nv12, int src_width,
                                         int src_height, int src_y_stride,
                                         int src_uv_stride, int roi_x,
                                         int roi_y, int roi_w, int roi_h,
                                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
