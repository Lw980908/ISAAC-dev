#pragma once
#include"common_include.h"
#include"yolo_utils.h"

#define CHECK(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

#define BLOCK_SIZE 8

//note: resize rgb with padding
void resizeDevice(const int& batch_size, float* src, int src_width, int src_height,
    float* dst, int dstWidth, int dstHeight,
    float paddingValue, yolo_utils::AffineMat matrix, cudaStream_t stream = 0);

//overload:resize rgb with padding, but src's type is uin8
void resizeDevice(const int& batch_size, unsigned char* src, int src_width, int src_height,
    float* dst, int dstWidth, int dstHeight,
    float paddingValue, yolo_utils::AffineMat matrix, cudaStream_t stream = 0);

// overload: resize rgb/gray without padding
void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight,
    yolo_utils::ColorMode mode, yolo_utils::AffineMat matrix, cudaStream_t stream = 0);

void bgr2rgbDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight, cudaStream_t stream = 0);

void normDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight,
    InitParameter norm_param, cudaStream_t stream = 0);

void hwc2chwDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight, cudaStream_t stream = 0);

void decodeDevice(InitParameter param, float* src, int srcWidth, int srcHeight, int srcLength, float* dst, int dstWidth, int dstHeight, cudaStream_t stream = 0);

// nms fast
void nmsDeviceV1(InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, cudaStream_t stream = 0);

// nms sort
void nmsDeviceV2(InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea,
    int* idx, float* conf, cudaStream_t stream = 0);

void copyWithPaddingDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight, float paddingValue, int padTop, int padLeft, cudaStream_t stream = 0);