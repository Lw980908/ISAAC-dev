cmake_minimum_required(VERSION 3.10)
project(isaac_ros_cam_and_lidar VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../isaac_ros_object_detection/src/cmake/common.cmake)

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(livox_ros_driver2 QUIET)
find_package(object_detection QUIET)
find_package(isaac_ros_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(COLORED_CLOUD_VIEW_RENDERER_SOURCES
  src/colored_cloud_view_renderer.cpp
  ../isaac_ros_object_detection/src/perception/gxf_nv12_convert.cu
)

set(PROJECTION_DEMO_SOURCES
  src/camera_lidar_projection_demo.cpp
  ../isaac_ros_object_detection/src/perception/gxf_nv12_convert.cu
  ../isaac_ros_object_detection/src/perception/gxf_nv12_draw.cu
)

set(CAMERA_COMBINED_WITH_LIDAR_COMPONENT_SOURCES
  src/camera_combined_with_lidar_component.cpp
  src/camera_combined_with_lidar.cpp
  src/lidar_measure_distance.cpp
)

set(INSTALL_TARGETS)

add_library(camera_lidar_projection_demo_component SHARED
  ${PROJECTION_DEMO_SOURCES}
)

set_target_properties(camera_lidar_projection_demo_component PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
  PREFIX "lib"
)

target_include_directories(camera_lidar_projection_demo_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(camera_lidar_projection_demo_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cv_bridge
  isaac_ros_nitros
  isaac_ros_nitros_image_type
  isaac_ros_managed_nitros
)

if(livox_ros_driver2_FOUND)
  message(STATUS
    "Found livox_ros_driver2 via CMAKE_PREFIX_PATH/AMENT_PREFIX_PATH: "
    "${livox_ros_driver2_DIR}")
  ament_target_dependencies(camera_lidar_projection_demo_component
    livox_ros_driver2
  )
else()
  message(WARNING
    "livox_ros_driver2 was not found. camera_lidar_projection_demo will "
    "subscribe to sensor_msgs/msg/PointCloud2 only. "
    "To enable livox_ros_driver2/msg/CustomMsg support, source the workspace "
    "that installed livox_ros_driver2 before building this package, or pass "
    "its prefix through CMAKE_PREFIX_PATH/livox_ros_driver2_DIR.")
endif()

target_link_libraries(camera_lidar_projection_demo_component
  ${OpenCV_LIBS}
  CUDA::cudart
)

rclcpp_components_register_nodes(
  camera_lidar_projection_demo_component
  "CameraLidarProjectionDemo"
)

add_executable(camera_lidar_projection_demo
  ${PROJECTION_DEMO_SOURCES}
)

set_target_properties(camera_lidar_projection_demo PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_include_directories(camera_lidar_projection_demo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(camera_lidar_projection_demo
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cv_bridge
  isaac_ros_nitros
  isaac_ros_nitros_image_type
  isaac_ros_managed_nitros
)

if(livox_ros_driver2_FOUND)
  ament_target_dependencies(camera_lidar_projection_demo
    livox_ros_driver2
  )
endif()

target_link_libraries(camera_lidar_projection_demo
  ${OpenCV_LIBS}
  CUDA::cudart
)

list(APPEND INSTALL_TARGETS
  camera_lidar_projection_demo
  camera_lidar_projection_demo_component
)

if(livox_ros_driver2_FOUND AND object_detection_FOUND)
  set(OBJECT_DETECTION_LINK_LIBS)
  if(TARGET object_detection)
    list(APPEND OBJECT_DETECTION_LINK_LIBS object_detection)
  elseif(DEFINED object_detection_LIBRARIES)
    list(APPEND OBJECT_DETECTION_LINK_LIBS ${object_detection_LIBRARIES})
  endif()

  add_library(camera_combined_with_lidar_component SHARED
    ${CAMERA_COMBINED_WITH_LIDAR_COMPONENT_SOURCES}
  )

  set_target_properties(camera_combined_with_lidar_component PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    PREFIX "lib"
  )

  target_include_directories(camera_combined_with_lidar_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../isaac_ros_object_detection/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../isaac_ros_object_detection/src/perception>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${INCLUDE_DRIS}
  )

  ament_target_dependencies(camera_combined_with_lidar_component
    rclcpp
    rclcpp_components
    sensor_msgs
    std_msgs
    visualization_msgs
    cv_bridge
    livox_ros_driver2
    isaac_ros_nitros
    isaac_ros_nitros_image_type
    isaac_ros_managed_nitros
  )

  target_link_libraries(camera_combined_with_lidar_component
    ${OBJECT_DETECTION_LINK_LIBS}
    ${OpenCV_LIBS}
    CUDA::cudart
  )

  rclcpp_components_register_nodes(
    camera_combined_with_lidar_component
    "isaac_ros_cam_and_lidar::CameraCombinedWithLidarComponent"
  )

  list(APPEND INSTALL_TARGETS
    camera_combined_with_lidar_component
  )
else()
  message(WARNING
    "camera_combined_with_lidar_component was not built because "
    "livox_ros_driver2 or object_detection was not found.")
endif()

add_library(colored_cloud_view_renderer_component SHARED
  ${COLORED_CLOUD_VIEW_RENDERER_SOURCES}
)

set_target_properties(colored_cloud_view_renderer_component PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
  PREFIX "lib"
)

target_include_directories(colored_cloud_view_renderer_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(colored_cloud_view_renderer_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  isaac_ros_nitros
  isaac_ros_nitros_image_type
  isaac_ros_managed_nitros
)

target_link_libraries(colored_cloud_view_renderer_component
  ${OpenCV_LIBS}
  CUDA::cudart
)

rclcpp_components_register_nodes(
  colored_cloud_view_renderer_component
  "ColoredCloudViewRenderer"
)

add_executable(colored_cloud_view_renderer
  ${COLORED_CLOUD_VIEW_RENDERER_SOURCES}
)

set_target_properties(colored_cloud_view_renderer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_include_directories(colored_cloud_view_renderer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(colored_cloud_view_renderer
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  isaac_ros_nitros
  isaac_ros_nitros_image_type
  isaac_ros_managed_nitros
)

target_link_libraries(colored_cloud_view_renderer
  ${OpenCV_LIBS}
  CUDA::cudart
)

list(APPEND INSTALL_TARGETS
  colored_cloud_view_renderer
  colored_cloud_view_renderer_component
)

install(TARGETS
  ${INSTALL_TARGETS}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
