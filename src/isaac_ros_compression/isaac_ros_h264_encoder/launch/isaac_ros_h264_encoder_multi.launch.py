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
        get_package_share_directory('isaac_ros_h264_encoder'),
        'config',
        'h264_encoder_multi.yaml')
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

        encoders = params_data.get('encoders', [])
        if not isinstance(encoders, list):
            raise RuntimeError('YAML "encoders" must be a list')

        encoder_nodes = []
        for index, encoder in enumerate(encoders):
            if not isinstance(encoder, dict):
                raise RuntimeError(f'encoder[{index}] must be a map')

            name = encoder.get('name', f'h264_encoder_{index}')
            namespace = encoder.get('namespace', '')
            params = {
                'input_width': encoder.get('input_width', 1920),
                'input_height': encoder.get('input_height', 1200),
                'qp': encoder.get('qp', 20),
                'hw_preset_type': encoder.get('hw_preset_type', 0),
                'profile': encoder.get('profile', 0),
                'iframe_interval': encoder.get('iframe_interval', 5),
                'config': encoder.get('config', 'pframe_cqp'),
                # 可选：设置输入协商的优先格式（不限制协商范围）
                # 例如："nitros_image_nv12" 或 "nitros_image_bgr8"
                'compatible_input_format': encoder.get(
                    'compatible_input_format', 'nitros_image_nv12'),
            }

            input_topic = encoder.get('input_topic', 'image_raw')
            output_topic = encoder.get('output_topic', 'image_compressed')
            remappings = []
            if input_topic != 'image_raw':
                remappings.append(('image_raw', input_topic))
            if output_topic != 'image_compressed':
                remappings.append(('image_compressed', output_topic))

            encoder_nodes.append(
                ComposableNode(
                    name=name,
                    namespace=namespace,
                    package='isaac_ros_h264_encoder',
                    plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
                    parameters=[params],
                    remappings=remappings,
                )
            )

        if not encoder_nodes:
            raise RuntimeError('No encoders configured in YAML')

        container = ComposableNodeContainer(
            name='h264_encoder_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=encoder_nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(
        function=launch_setup)])
