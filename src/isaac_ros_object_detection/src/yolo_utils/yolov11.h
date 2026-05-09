#pragma once
#include "yolo.h"
#include "yolo_utils.h"
class YOLOV11 : public yolo::YOLO
{
public:
	YOLOV11() = default;				 // 默认构造函数
	YOLOV11(const InitParameter &param); // 带参数的构造函数
	~YOLOV11();
	// 重要：避免派生类重载导致基类同名函数被隐藏（例如 YOLO::preprocess() 无参版）
	using yolo::YOLO::preprocess;
	virtual bool init(const std::vector<unsigned char> &trtFile);
	virtual void preprocess(const std::vector<cv::Mat> &imgsBatch);
	virtual void postprocess(const std::vector<cv::Mat> &imgsBatch);
	virtual void postprocess(int batch_size);

private:
	float *m_output_src_transpose_device;
};