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

import os
import yaml

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('isaac_ros_v4l2_camera'),
        'config',
        'v4l2_camera_multi.yaml')
    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to YAML parameters file.'),
    ]

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)
        with open(params_path, 'r', encoding='utf-8') as handle:
            params_data = yaml.safe_load(handle) or {}

        cameras = params_data.get('cameras', [])
        if not isinstance(cameras, list):
            raise RuntimeError('YAML "cameras" must be a list')

        camera_nodes = []
        for index, camera in enumerate(cameras):
            if not isinstance(camera, dict):
                raise RuntimeError(f'camera[{index}] must be a map')
            name = camera.get('name')
            namespace = camera.get('namespace', '')
            if not name:
                raise RuntimeError(f'camera[{index}] missing "name"')

            params = {
                'device': camera.get('device', f'/dev/video{index}'),
                'camera_info_url': camera.get('camera_info_url', ''),
                'camera_link_frame_name': camera.get(
                    'camera_link_frame_name', f'camera_{index}'),
                'optical_frame_name': camera.get(
                    'optical_frame_name', f'camera_{index}_optical'),
                'image_topic': camera.get('image_topic', 'image_raw'),
                'camera_info_topic': camera.get('camera_info_topic',
                                                'camera_info'),
                'gpu_id': camera.get('gpu_id', 0),
                'width': camera.get('width', 1920),
                'height': camera.get('height', 1536),
                'framerate': camera.get('framerate', 30),
                'pixel_format': camera.get('pixel_format', 'YUYV'),
                'output_encoding': camera.get('output_encoding', 'nv12'),
                'buffer_count': camera.get('buffer_count', 2),
            }

            camera_nodes.append(
                ComposableNode(
                    name=name,
                    package='isaac_ros_v4l2_camera',
                    plugin='nvidia::isaac_ros::v4l2_camera::V4l2GpuCameraNode',
                    namespace=namespace,
                    parameters=[params],
                )
            )

        if not camera_nodes:
            raise RuntimeError('No cameras configured in YAML')

        v4l2_container = ComposableNodeContainer(
            name='v4l2_gpu_camera_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=camera_nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [v4l2_container]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(
        function=launch_setup)])
