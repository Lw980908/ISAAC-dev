set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.10)

# 设置 CUDA 架构（在 project() 之前设置，避免 CMP0104 警告）
# Jetson Orin: 87, Jetson Xavier: 72, Jetson Nano: 53
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # Jetson Orin (JetPack 5.x)
    set(CMAKE_CUDA_ARCHITECTURES 87)
else()
    # 桌面 GPU（如 RTX 30 系列 = 86）
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(object_detection VERSION 1.0.0 LANGUAGES C CXX CUDA)

# 检测 Jetson 平台
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_definitions(-DIS_JETSON)
    message(STATUS "Building for Jetson platform (ARM64)")
else()
    message(STATUS "Building for x86_64 platform")
endif()

find_package(ament_cmake REQUIRED)

# 抑制弃用警告
add_compile_options(-Wno-deprecated-declarations)

# 添加 CUDA 和 TensorRT 的库文件
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/common.cmake)

# ===== TensorRT 版本检测与兼容性处理 =====
# TensorRT 8.x 和 10.x 的 samples/common 目录结构不同
# 通过检测特定文件来判断版本

set(TENSORRT_COMPAT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tensorrt_compat)

# 检测 TensorRT 版本（通过检查 utils/timingCache.h 是否存在）
if(EXISTS "${TensorRT_ROOT}/samples/common/utils/timingCache.h")
    # TensorRT 8.x - 系统自带 timingCache.h
    set(TENSORRT_VERSION_8X TRUE)
    message(STATUS "TensorRT 8.x detected - using system samples headers")
    set(TENSORRT_COMPAT_INCLUDE "")
else()
    # TensorRT 10.x - 需要使用兼容层
    set(TENSORRT_VERSION_8X FALSE)
    message(STATUS "TensorRT 10.x detected - using compatibility layer")
    set(TENSORRT_COMPAT_INCLUDE ${TENSORRT_COMPAT_DIR})
endif()

# 检测 bfloat16.cpp 是否存在（TensorRT 10.x 才有）
if(EXISTS "${TensorRT_ROOT}/samples/common/bfloat16.cpp")
    set(TENSORRT_HAS_BFLOAT16 TRUE)
    message(STATUS "TensorRT bfloat16 support detected")
else()
    set(TENSORRT_HAS_BFLOAT16 FALSE)
    message(STATUS "TensorRT bfloat16 support NOT detected")
endif()
# ===== TensorRT 版本检测结束 =====

# 添加 OpenCV、MQTT 和 JSON 库
find_package(rclcpp_components REQUIRED)
find_package(OpenCV REQUIRED)

# 检测 OpenCV CUDA 模块是否可用
if(OpenCV_CUDA_VERSION OR "cudaarithm" IN_LIST OpenCV_LIB_COMPONENTS)
    message(STATUS "OpenCV CUDA module found - GPU undistortion enabled")
    add_definitions(-DHAVE_OPENCV_CUDA)
else()
    # 尝试查找 CUDA 相关的 OpenCV 库
    find_library(OPENCV_CUDA_LIB opencv_cudaarithm HINTS ${OpenCV_INSTALL_PATH}/lib)
    if(OPENCV_CUDA_LIB)
        message(STATUS "OpenCV CUDA library found: ${OPENCV_CUDA_LIB}")
        add_definitions(-DHAVE_OPENCV_CUDA)
    else()
        message(STATUS "OpenCV CUDA module NOT found - GPU undistortion will fallback to CPU")
    endif()
endif()

find_package(PahoMqttCpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_library(PahoMqttCpp_LIB NAMES paho-mqttpp3 PATHS /home/nvidia/paho.mqtt.cpp/build/src)
find_library(PahoMqtt3As_LIB NAMES paho-mqtt3as PATHS /home/nvidia/paho.mqtt.cpp/build/src)
link_directories(/home/nvidia/paho.mqtt.cpp/build/src)

# 查找 ROS2 库
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(class_loader REQUIRED)
find_package(console_bridge REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

# NITROS 相关依赖（用于与 H264 编码器配合的协商话题格式）
find_package(isaac_ros_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)

# VPI (NVIDIA Vision Programming Interface) 用于 NV12 <-> BGR 图像格式转换
find_package(vpi REQUIRED)

# 添加源文件
file(GLOB CPPS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/perception/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/perception/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_utils/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/distortion_correction/undistort_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/distortion_correction/undistort_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/distortion_correction/undistorter_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/distortion_correction/undistort_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mqtt/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MobileNetV2/classify.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ros2/ros2_client.cpp 
    ${TensorRT_ROOT}/samples/common/logger.cpp 
    ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
    ${TensorRT_ROOT}/samples/common/sampleUtils.cpp 
    ${TensorRT_ROOT}/samples/common/plugin.cpp 
)

# 避免 GLOB 把“误拷贝/误命名”的重复源文件也编译进来，造成链接期 multiple definition：
# - 入口文件/组件包装文件由各自 target 单独添加，不应进入通用源文件集合
# - 排除文件名包含 '&' 的 .cpp（常见于误保存产生的重复文件）
list(FILTER CPPS EXCLUDE REGEX ".*/src/perception/main\\.cpp$")
list(FILTER CPPS EXCLUDE REGEX ".*/src/perception/perception_component\\.cpp$")
list(FILTER CPPS EXCLUDE REGEX ".*/src/perception/.*&.*\\.cpp$")

# TensorRT 10.x 需要 bfloat16.cpp
if(TENSORRT_HAS_BFLOAT16)
    list(APPEND CPPS ${TensorRT_ROOT}/samples/common/bfloat16.cpp)
endif()

# 添加包含路径
# 根据 TensorRT 版本动态设置 include 路径
if(TENSORRT_VERSION_8X)
    # TensorRT 8.x - 使用系统自带的 samples 头文件
    include_directories(
        ${CUDA_INCLUDE_DIRS} 
        ${OpenCV_INCLUDE_DIRS} 
        ${TENSORRT_INCLUDE_DIR} 
        ${TensorRT_ROOT}/samples/common
        ${TensorRT_ROOT}/samples/common/utils  # TensorRT 8.x utils 目录
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${INCLUDE_DRIS}
    )
else()
    # TensorRT 10.x - 使用兼容层（优先于系统路径）
    include_directories(
        ${TENSORRT_COMPAT_INCLUDE}  # TensorRT 版本兼容层（优先）
        ${CUDA_INCLUDE_DIRS} 
        ${OpenCV_INCLUDE_DIRS} 
        ${TENSORRT_INCLUDE_DIR} 
        ${TensorRT_ROOT}/samples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${INCLUDE_DRIS}
    )
endif()

# 添加 cudnn 和 cublas
find_library(CUDNN_LIB cudnn HINTS /usr/local/cuda/lib64)
find_library(CUBLAS_LIB cublas HINTS /usr/local/cuda/lib64)
list(APPEND ALL_LIBS ${CUDNN_LIB} ${CUBLAS_LIB})

# 查找V4L2库
find_library(V4L2_LIB v4l2)

# 创建共享库
add_library(${PROJECT_NAME} SHARED ${CPPS})

# 添加 ROS 2 依赖
ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    rclcpp_components
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    livox_ros_driver2
    cv_bridge
    OpenCV
    nlohmann_json
    class_loader
    console_bridge
    Boost
    isaac_ros_nitros
    isaac_ros_nitros_image_type
    isaac_ros_managed_nitros
    vpi
)

# 链接所有依赖
target_link_libraries(${PROJECT_NAME} 
    ${ALL_LIBS} 
    ${OpenCV_LIBS}
    ${PahoMqttCpp_LIB} 
    ${PahoMqtt3As_LIB} 
    nlohmann_json::nlohmann_json 
    Boost::filesystem 
    Boost::system
    m
    class_loader::class_loader
    console_bridge::console_bridge 
)

add_executable(object_detection_node 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/perception/main.cpp 
)

# object_detection_node 需要 ROS 2 依赖
ament_target_dependencies(object_detection_node
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    livox_ros_driver2
    cv_bridge
)

# 创建可执行文件并链接共享库
target_link_libraries(object_detection_node 
    ${PROJECT_NAME}
)

# ===== 组件库定义 =====
# 创建组件共享库
add_library(perception_component SHARED
    src/perception/perception_component.cpp
    ${CPPS}
)

# 设置包含目录
target_include_directories(perception_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

# 链接依赖
ament_target_dependencies(perception_component
    rclcpp
    rclcpp_components
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    livox_ros_driver2
    cv_bridge
    OpenCV
    nlohmann_json
    class_loader
    console_bridge
    Boost
    isaac_ros_nitros
    isaac_ros_nitros_image_type
    isaac_ros_managed_nitros
    vpi
)

# 链接其他依赖
target_link_libraries(perception_component
    ${ALL_LIBS} 
    ${OpenCV_LIBS}
    ${PahoMqttCpp_LIB} 
    ${PahoMqtt3As_LIB} 
    nlohmann_json::nlohmann_json 
    Boost::filesystem 
    Boost::system
    m
    v4l2
    class_loader::class_loader
    console_bridge::console_bridge
    ${class_loader_LIBRARIES}
    ${console_bridge_LIBRARIES}
)

# 注册组件
rclcpp_components_register_nodes(perception_component "object_detection::PerceptionComponent")
# ===== 组件库定义结束 =====

# 添加编译选项
target_compile_options(object_detection_node PRIVATE 
    -Wno-deprecated-declarations 
    -lineinfo 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --default-stream per-thread 
        -lineinfo 
        --use_fast_math 
        --disable-warnings>
)

# 设置RPATH
set_target_properties(object_detection_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

set_target_properties(perception_component PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# 安装所有目标
install(TARGETS ${PROJECT_NAME} perception_component object_detection_node
    EXPORT ${PROJECT_NAME}_exports
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/
)

# 导出组件
ament_export_libraries(perception_component ${PROJECT_NAME})
ament_export_dependencies(
    rclcpp_components
    class_loader
    console_bridge
    OpenCV
    cv_bridge
    sensor_msgs
    geometry_msgs
    visualization_msgs
    nlohmann_json
    Boost
    isaac_ros_nitros
    isaac_ros_nitros_image_type
    isaac_ros_managed_nitros
    vpi
)

ament_package()
