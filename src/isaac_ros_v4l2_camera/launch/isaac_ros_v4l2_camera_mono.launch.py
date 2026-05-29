# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'device',
            default_value='/dev/video0',
            description='V4L2 device path.'),
        DeclareLaunchArgument(
            'camera_info_url',
            default_value='',
            description='URL for the camera info file.'),
        DeclareLaunchArgument(
            'output_encoding',
            default_value='nv12',
            description='Output encoding / negotiated type: nv12 or bgr8.'),
    ]

    device = LaunchConfiguration('device')
    camera_info_url = LaunchConfiguration('camera_info_url')
    output_encoding = LaunchConfiguration('output_encoding')

    v4l2_node = ComposableNode(
        name='v4l2_gpu_camera',
        package='isaac_ros_v4l2_camera',
        plugin='nvidia::isaac_ros::v4l2_camera::V4l2GpuCameraNode',
        namespace='',
        parameters=[{
            'device': device,
            'camera_info_url': camera_info_url,
            'output_encoding': output_encoding,
        }],
    )

    v4l2_container = ComposableNodeContainer(
        name='v4l2_gpu_camera_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[v4l2_node],
        namespace='',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return launch.LaunchDescription(launch_args + [v4l2_container])