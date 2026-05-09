set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
# find thirdparty
find_package(CUDA REQUIRED)
list(APPEND ALL_LIBS 
  ${CUDA_LIBRARIES} 
  ${CUDA_cublas_LIBRARY} 
  ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY})

# include cuda's header
list(APPEND INCLUDE_DRIS ${CUDA_INCLUDE_DIRS})
# message(FATAL_ERROR "CUDA_npp_LIBRARY: ${CUDA_npp_LIBRARY}")

# gather TensorRT lib
# set(TensorRT_ROOT /home/alienware/TensorRT-8.6.1.6)
set(TensorRT_ROOT /usr/src/tensorrt)

# ============================================================================
# TensorRT 版本检测
# ============================================================================
# 尝试从 NvInferVersion.h 检测 TensorRT 版本
find_path(TENSORRT_VERSION_INCLUDE_DIR NAMES NvInferVersion.h 
    HINTS ${TensorRT_ROOT} /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include
    PATH_SUFFIXES include)

if(TENSORRT_VERSION_INCLUDE_DIR)
    file(READ "${TENSORRT_VERSION_INCLUDE_DIR}/NvInferVersion.h" TRT_VERSION_FILE)
    
    # 提取主版本号
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)" _ ${TRT_VERSION_FILE})
    set(TRT_VERSION_MAJOR ${CMAKE_MATCH_1})
    
    # 提取次版本号
    string(REGEX MATCH "#define NV_TENSORRT_MINOR ([0-9]+)" _ ${TRT_VERSION_FILE})
    set(TRT_VERSION_MINOR ${CMAKE_MATCH_1})
    
    # 提取补丁版本号
    string(REGEX MATCH "#define NV_TENSORRT_PATCH ([0-9]+)" _ ${TRT_VERSION_FILE})
    set(TRT_VERSION_PATCH ${CMAKE_MATCH_1})
    
    set(TRT_VERSION "${TRT_VERSION_MAJOR}.${TRT_VERSION_MINOR}.${TRT_VERSION_PATCH}")
    message(STATUS "===========================================")
    message(STATUS "TensorRT Version: ${TRT_VERSION}")
    message(STATUS "  Major: ${TRT_VERSION_MAJOR}")
    message(STATUS "  Minor: ${TRT_VERSION_MINOR}")  
    message(STATUS "  Patch: ${TRT_VERSION_PATCH}")
    
    if(TRT_VERSION_MAJOR GREATER_EQUAL 10)
        message(STATUS "Using TensorRT 10.x compatible API")
    else()
        message(STATUS "Using TensorRT 8.x compatible API")
    endif()
    message(STATUS "===========================================")
else()
    message(WARNING "Could not find NvInferVersion.h to detect TensorRT version")
endif()

find_library(TRT_NVINFER NAMES nvinfer HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_NVONNX_PARSER NAMES nvonnxparser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)

# nvcaffe_parser 在 TensorRT 8.x+ 中已移除，设为可选
find_library(TRT_NVCAFFE_PARSER NAMES nvcaffe_parser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
if(TRT_NVCAFFE_PARSER)
    list(APPEND ALL_LIBS ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN} ${TRT_NVONNX_PARSER} ${TRT_NVCAFFE_PARSER})
    message(STATUS "Found nvcaffe_parser: ${TRT_NVCAFFE_PARSER}")
else()
    list(APPEND ALL_LIBS ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN} ${TRT_NVONNX_PARSER})
    message(STATUS "nvcaffe_parser not found (optional, skipped)")
endif()

# include tensorrt's headers
list(APPEND INCLUDE_DRIS ${TENSORRT_INCLUDE_DIR})

# include tensorrt's sample/common headers
#set(SAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)
#set(SAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(SAMPLES_COMMON_DIR ${TensorRT_ROOT}/samples/common)
list(APPEND INCLUDE_DRIS ${SAMPLES_COMMON_DIR})

# TensorRT 8.6+ 需要 utils 子目录
if(EXISTS "${SAMPLES_COMMON_DIR}/utils")
    list(APPEND INCLUDE_DRIS ${SAMPLES_COMMON_DIR}/utils)
    message(STATUS "Found TensorRT samples/common/utils directory")
endif()

message(STATUS ***INCLUDE_DRIS*** = ${INCLUDE_DRIS})
message(STATUS "ALL_LIBS: ${ALL_LIBS}")
