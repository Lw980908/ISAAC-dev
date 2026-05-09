cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_v4l2_camera LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

find_package(ament_cmake_auto REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
ament_auto_find_build_dependencies()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(nvbufsurface SHARED IMPORTED)
add_library(nvbufsurftransform SHARED IMPORTED)

if(${ARCHITECTURE} STREQUAL "x86_64")
  set_property(TARGET nvbufsurface PROPERTY IMPORTED_LOCATION
    /usr/lib/x86_64-linux-gnu/libnvbufsurface.so)
  set_property(TARGET nvbufsurftransform PROPERTY IMPORTED_LOCATION
    /usr/lib/x86_64-linux-gnu/libnvbufsurftransform.so)
elseif(${ARCHITECTURE} STREQUAL "aarch64")
  set_property(TARGET nvbufsurface PROPERTY IMPORTED_LOCATION
    /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so)
  set_property(TARGET nvbufsurftransform PROPERTY IMPORTED_LOCATION
    /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurftransform.so)
endif()


ament_auto_add_library(
  v4l2_gpu_camera_node
  SHARED
  src/v4l2_gpu_camera_node.cpp
  src/nv12_format_convert.cu
)

set_target_properties(v4l2_gpu_camera_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(v4l2_gpu_camera_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../isaac_ros_compression/gxf/codec/nvbuf_utils
)

target_link_libraries(v4l2_gpu_camera_node
  CUDA::cuda_driver
  CUDA::cudart
  nvbufsurface
  nvbufsurftransform
  EGL
)

rclcpp_components_register_nodes(
  v4l2_gpu_camera_node "nvidia::isaac_ros::v4l2_camera::V4l2GpuCameraNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::v4l2_camera::V4l2GpuCameraNode;$<TARGET_FILE:v4l2_gpu_camera_node>\n")

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat
  libavcodec
  libavutil
  libswscale
)

ament_auto_add_library(
  mp4_nv12_nitros_publisher_node
  SHARED
  src/mp4_nv12_nitros_publisher_node.cpp
)

target_include_directories(mp4_nv12_nitros_publisher_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mp4_nv12_nitros_publisher_node
  CUDA::cudart
  PkgConfig::FFMPEG
)

rclcpp_components_register_nodes(
  mp4_nv12_nitros_publisher_node
  "nvidia::isaac_ros::v4l2_camera::Mp4Nv12NitrosPublisherNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::v4l2_camera::Mp4Nv12NitrosPublisherNode;$<TARGET_FILE:mp4_nv12_nitros_publisher_node>\n")

ament_export_include_directories(include)
set(install_share_dirs config)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  list(APPEND install_share_dirs launch)
endif()
ament_auto_package(INSTALL_TO_SHARE ${install_share_dirs})
