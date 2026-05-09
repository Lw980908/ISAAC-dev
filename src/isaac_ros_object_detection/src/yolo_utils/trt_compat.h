/**
 * @file trt_compat.h
 * @brief TensorRT 8.x / 10.x 兼容性层
 * 
 * 该文件提供了 TensorRT 8.x 和 10.x 之间的 API 兼容性宏和辅助函数。
 * 主要变化：
 * - TRT 10.x 移除了 getBindingDimensions()，改用 getTensorShape()
 * - TRT 10.x 移除了 setBindingDimensions()，改用 setInputShape()
 * - TRT 10.x 移除了 getBindingIndex()，改用 基于名称的 API
 * - TRT 10.x 的 deserializeCudaEngine() 只需要 2 个参数
 * - TRT 10.x 移除了 destroy() 方法，使用 delete 代替
 * - TRT 10.x 使用 enqueueV3() 和基于名称的张量地址设置
 */

#ifndef TRT_COMPAT_H
#define TRT_COMPAT_H

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
// TensorRT 版本检测
// ============================================================================

#ifndef NV_TENSORRT_MAJOR
#error "NV_TENSORRT_MAJOR not defined. Please include NvInferVersion.h"
#endif

// TensorRT 10.x 或更高版本
#if NV_TENSORRT_MAJOR >= 10
    #define TRT_VERSION_10_OR_LATER 1
#else
    #define TRT_VERSION_10_OR_LATER 0
#endif

// ============================================================================
// 兼容性宏定义
// ============================================================================

/**
 * @brief 反序列化 CUDA 引擎
 * TRT 8.x: deserializeCudaEngine(data, size, nullptr)
 * TRT 10.x: deserializeCudaEngine(data, size)
 */
#if TRT_VERSION_10_OR_LATER
    #define TRT_DESERIALIZE_ENGINE(runtime, data, size) \
        (runtime)->deserializeCudaEngine((data), (size))
#else
    #define TRT_DESERIALIZE_ENGINE(runtime, data, size) \
        (runtime)->deserializeCudaEngine((data), (size), nullptr)
#endif

/**
 * @brief 获取张量维度（通过索引）
 * TRT 8.x: getBindingDimensions(index)
 * TRT 10.x: getTensorShape(getIOTensorName(index))
 */
#if TRT_VERSION_10_OR_LATER
    #define TRT_GET_BINDING_DIMS(engine, index) \
        (engine)->getTensorShape((engine)->getIOTensorName(index))
    
    #define TRT_CONTEXT_GET_BINDING_DIMS(context, engine, index) \
        (context)->getTensorShape((engine)->getIOTensorName(index))
#else
    #define TRT_GET_BINDING_DIMS(engine, index) \
        (engine)->getBindingDimensions(index)
    
    #define TRT_CONTEXT_GET_BINDING_DIMS(context, engine, index) \
        (context)->getBindingDimensions(index)
#endif

/**
 * @brief 设置输入张量维度（动态 batch）
 * TRT 8.x: setBindingDimensions(index, dims)
 * TRT 10.x: setInputShape(name, dims)
 */
#if TRT_VERSION_10_OR_LATER
    #define TRT_SET_INPUT_SHAPE(context, engine, index, dims) \
        (context)->setInputShape((engine)->getIOTensorName(index), (dims))
#else
    #define TRT_SET_INPUT_SHAPE(context, engine, index, dims) \
        (context)->setBindingDimensions((index), (dims))
#endif

/**
 * @brief 获取绑定索引（通过名称）
 * TRT 8.x: getBindingIndex(name)
 * TRT 10.x: 需要遍历 IO 张量名称
 */
#if TRT_VERSION_10_OR_LATER
    // TRT 10.x 不再有 getBindingIndex，需要手动查找
    inline int trt_get_binding_index(nvinfer1::ICudaEngine* engine, const char* name)
    {
        int num_io = engine->getNbIOTensors();
        for (int i = 0; i < num_io; ++i)
        {
            if (std::string(engine->getIOTensorName(i)) == name)
            {
                return i;
            }
        }
        return -1;
    }
    #define TRT_GET_BINDING_INDEX(engine, name) trt_get_binding_index((engine), (name))
#else
    #define TRT_GET_BINDING_INDEX(engine, name) (engine)->getBindingIndex(name)
#endif

// ============================================================================
// 推理执行兼容性
// ============================================================================

/**
 * @brief 执行推理
 * TRT 8.x: executeV2(bindings)
 * TRT 10.x: 需要使用 setTensorAddress + enqueueV3
 */
#if TRT_VERSION_10_OR_LATER

// TRT 10.x 推理辅助类
class TrtInferHelper
{
public:
    /**
     * @brief 执行推理（兼容 TRT 10.x）
     * @param context 执行上下文
     * @param engine 引擎
     * @param bindings 绑定数组（按索引顺序）
     * @param stream CUDA 流（可选，默认为 0）
     * @return 是否成功
     */
    static bool execute(nvinfer1::IExecutionContext* context, 
                       nvinfer1::ICudaEngine* engine,
                       void** bindings, 
                       cudaStream_t stream = 0)
    {
        int num_io = engine->getNbIOTensors();
        for (int i = 0; i < num_io; ++i)
        {
            const char* name = engine->getIOTensorName(i);
            if (!context->setTensorAddress(name, bindings[i]))
            {
                return false;
            }
        }
        return context->enqueueV3(stream);
    }
    
    /**
     * @brief 同步执行推理（兼容 TRT 10.x）
     */
    static bool executeSync(nvinfer1::IExecutionContext* context,
                           nvinfer1::ICudaEngine* engine,
                           void** bindings)
    {
        int num_io = engine->getNbIOTensors();
        for (int i = 0; i < num_io; ++i)
        {
            const char* name = engine->getIOTensorName(i);
            if (!context->setTensorAddress(name, bindings[i]))
            {
                return false;
            }
        }
        // enqueueV3 with stream 0 and sync
        bool result = context->enqueueV3(0);
        cudaStreamSynchronize(0);
        return result;
    }
};

    #define TRT_EXECUTE_V2(context, engine, bindings) \
        TrtInferHelper::execute((context), (engine), (void**)(bindings), 0)
        
    #define TRT_EXECUTE_V2_ASYNC(context, engine, bindings, stream) \
        TrtInferHelper::execute((context), (engine), (void**)(bindings), (stream))

#else
    // TRT 8.x 直接使用 executeV2
    #define TRT_EXECUTE_V2(context, engine, bindings) \
        (context)->executeV2((void**)(bindings))
        
    #define TRT_EXECUTE_V2_ASYNC(context, engine, bindings, stream) \
        (context)->enqueueV2((void**)(bindings), (stream), nullptr)
#endif

// ============================================================================
// 智能指针删除器
// ============================================================================

/**
 * @brief TensorRT 对象删除器
 * TRT 8.x: 调用 destroy()
 * TRT 10.x: 使用 delete
 */
#if TRT_VERSION_10_OR_LATER
    struct TrtDeleter
    {
        template<typename T>
        void operator()(T* obj) const
        {
            delete obj;
        }
    };
#else
    struct TrtDeleter
    {
        template<typename T>
        void operator()(T* obj) const
        {
            if (obj) obj->destroy();
        }
    };
#endif

// 使用自定义删除器的智能指针类型
using TrtRuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>;
using TrtEnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>;
using TrtContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>;

// 共享指针版本（带自定义删除器）
template<typename T>
using TrtSharedPtr = std::shared_ptr<T>;

template<typename T>
inline TrtSharedPtr<T> makeTrtShared(T* ptr)
{
#if TRT_VERSION_10_OR_LATER
    return TrtSharedPtr<T>(ptr, [](T* p) { delete p; });
#else
    return TrtSharedPtr<T>(ptr, [](T* p) { if(p) p->destroy(); });
#endif
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 获取 TensorRT 版本字符串
 */
inline std::string getTrtVersionString()
{
    return std::to_string(NV_TENSORRT_MAJOR) + "." + 
           std::to_string(NV_TENSORRT_MINOR) + "." + 
           std::to_string(NV_TENSORRT_PATCH);
}

/**
 * @brief 打印 TensorRT 版本信息
 */
inline void printTrtVersion()
{
    printf("TensorRT Version: %s\n", getTrtVersionString().c_str());
#if TRT_VERSION_10_OR_LATER
    printf("Using TensorRT 10.x compatible API\n");
#else
    printf("Using TensorRT 8.x compatible API\n");
#endif
}

#endif // TRT_COMPAT_H
