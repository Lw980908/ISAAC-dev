#pragma once
#include"yolo_utils.h"
#include"kernel_function.h"

namespace yolov11
{
	void decodeDevice(InitParameter param, float* src, int srcWidth, int srcHeight, int srcLength, float* dst, int dstWidth, int dstHeight, cudaStream_t stream = 0);
	void transposeDevice(InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight, cudaStream_t stream = 0);
}
