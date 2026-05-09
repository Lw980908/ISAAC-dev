# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
启动文件：MP4 多视频发布（NITROS NV12）
"""

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
        'v4l2_video_nv12.yaml')

    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to MP4 NV12 YAML parameters file.'),
    ]

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)

        with open(params_path, 'r', encoding='utf-8') as handle:
            params_data = yaml.safe_load(handle) or {}

        videos = params_data.get('videos', [])
        if not isinstance(videos, list):
            raise RuntimeError('YAML "videos" must be a list')
        if not videos:
            raise RuntimeError('No videos configured in YAML')

        video_nodes = []
        for index, video in enumerate(videos):
            if not isinstance(video, dict):
                raise RuntimeError(f'video[{index}] must be a map')
            name = video.get('name')
            namespace = video.get('namespace', '')
            if not name:
                raise RuntimeError(f'video[{index}] missing "name"')
            if not video.get('file_path'):
                raise RuntimeError(f'video[{index}] missing "file_path"')
            w = int(video.get('width', 0))
            h = int(video.get('height', 0))
            if w <= 0 or h <= 0:
                raise RuntimeError(
                    f'video[{index}] must set width/height (>0)')

            params = {
                'file_path': video.get('file_path'),
                'loop': bool(video.get('loop', True)),
                'gpu_id': int(video.get('gpu_id', 0)),
                'width': w,
                'height': h,
                'publish_fps': float(video.get('publish_fps', 0.0)),
                'image_topic': video.get('image_topic', 'image_raw'),
                'camera_info_topic': video.get('camera_info_topic', 'camera_info'),
                'camera_link_frame_name': video.get('camera_link_frame_name', f'camera_{index}'),
                'optical_frame_name': video.get('optical_frame_name', f'camera_{index}_optical'),
                'camera_info_url': video.get('camera_info_url', ''),
            }

            video_nodes.append(
                ComposableNode(
                    name=name,
                    package='isaac_ros_v4l2_camera',
                    plugin='nvidia::isaac_ros::v4l2_camera::Mp4Nv12NitrosPublisherNode',
                    namespace=namespace,
                    parameters=[params],
                )
            )

        container = ComposableNodeContainer(
            name='video_nv12_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=video_nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
