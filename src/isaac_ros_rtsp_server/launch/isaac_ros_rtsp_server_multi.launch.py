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
        get_package_share_directory('isaac_ros_rtsp_server'),
        'config',
        'rtsp_server_multi.yaml')
    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to RTSP server YAML parameters file.'),
    ]

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)
        with open(params_path, 'r', encoding='utf-8') as handle:
            params_data = yaml.safe_load(handle) or {}

        streams = params_data.get('streams', [])
        if not isinstance(streams, list):
            raise RuntimeError('YAML "streams" must be a list')

        nodes = []
        for index, stream in enumerate(streams):
            if not isinstance(stream, dict):
                raise RuntimeError(f'stream[{index}] must be a map')

            name = stream.get('name', f'rtsp_server_{index}')
            namespace = stream.get('namespace', '')
            use_nitros = stream.get('use_nitros', False)
            params = {
                'stream_name': stream.get('stream_name', name),
                'port': stream.get('port', 8554),
                'video_type': stream.get('video_type', 'h264'),
                'topic_name': stream.get('topic_name', 'image_compressed'),
                'assume_annexb': stream.get('assume_annexb', True),
                'dump_frame': stream.get('dump_frame', 0),
                'use_nitros': use_nitros,
                'nitros_format': stream.get(
                    'nitros_format', 'nitros_compressed_image'),
            }
            extra_arguments = []
            if not use_nitros:
                extra_arguments = [{'use_intra_process_comms': True}]

            nodes.append(
                ComposableNode(
                    name=name,
                    namespace=namespace,
                    package='isaac_ros_rtsp_server',
                    plugin='isaac_ros_rtsp_server::RtspServerNode',
                    parameters=[params],
                    extra_arguments=extra_arguments,
                )
            )

        if not nodes:
            raise RuntimeError('No streams configured in YAML')

        container = ComposableNodeContainer(
            name='rtsp_server_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(
        function=launch_setup)])
