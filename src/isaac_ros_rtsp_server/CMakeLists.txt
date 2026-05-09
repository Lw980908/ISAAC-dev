cmake_minimum_required(VERSION 3.5)
project(isaac_ros_rtsp_server)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(isaac_ros_nitros REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)
find_package(isaac_ros_nitros_compressed_image_type REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)
find_package(OpenSSL REQUIRED)

set(LIVE555_ROOT "" CACHE PATH "Path to live555 root directory")
if(LIVE555_ROOT STREQUAL "")
  set(LIVE555_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../live555")
endif()

set(LIVE555_INCLUDE_DIRS
  ${LIVE555_ROOT}/liveMedia/include
  ${LIVE555_ROOT}/groupsock/include
  ${LIVE555_ROOT}/UsageEnvironment/include
  ${LIVE555_ROOT}/BasicUsageEnvironment/include
  /usr/include/liveMedia
  /usr/include/groupsock
  /usr/include/UsageEnvironment
  /usr/include/BasicUsageEnvironment
  /usr/local/include/liveMedia
  /usr/local/include/groupsock
  /usr/local/include/UsageEnvironment
  /usr/local/include/BasicUsageEnvironment
)

set(LIVE555_LIB_DIRS
  ${LIVE555_ROOT}/liveMedia
  ${LIVE555_ROOT}/groupsock
  ${LIVE555_ROOT}/UsageEnvironment
  ${LIVE555_ROOT}/BasicUsageEnvironment
  /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
  /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
  /usr/local/lib
  /usr/lib
)

set(LIVE555_LIBS liveMedia groupsock UsageEnvironment BasicUsageEnvironment)

include_directories(
  include
  ${LIVE555_INCLUDE_DIRS}
)

set(RTSP_SERVER_SOURCES
  src/rtsp_component.cpp
  src/rtsp_server.cpp
  src/rtsp_server_node.cpp
  src/rtsp_server_component.cpp
  src/latency_monitor_node.cpp
  src/nitros_compressed_image_view.cpp
)

add_library(${PROJECT_NAME}_component SHARED ${RTSP_SERVER_SOURCES})
ament_target_dependencies(${PROJECT_NAME}_component
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  isaac_ros_nitros
  isaac_ros_managed_nitros
  isaac_ros_nitros_compressed_image_type
  isaac_ros_nitros_image_type
)
target_link_directories(${PROJECT_NAME}_component PRIVATE ${LIVE555_LIB_DIRS})
target_link_libraries(${PROJECT_NAME}_component
  ${LIVE555_LIBS}
  OpenSSL::SSL
  OpenSSL::Crypto
  pthread
  rt
  dl
)
set_target_properties(${PROJECT_NAME}_component PROPERTIES
  BUILD_RPATH "${LIVE555_LIB_DIRS}"
  INSTALL_RPATH "${LIVE555_LIB_DIRS}"
)

rclcpp_components_register_nodes(${PROJECT_NAME}_component
  "isaac_ros_rtsp_server::RtspServerNode"
  "isaac_ros_rtsp_server::LatencyMonitorNode"
)

add_executable(${PROJECT_NAME} src/main.cpp)
ament_target_dependencies(${PROJECT_NAME} rclcpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_component)
target_link_directories(${PROJECT_NAME} PRIVATE ${LIVE555_LIB_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${LIVE555_LIB_DIRS}"
  INSTALL_RPATH "${LIVE555_LIB_DIRS}"
)

install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_component
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

