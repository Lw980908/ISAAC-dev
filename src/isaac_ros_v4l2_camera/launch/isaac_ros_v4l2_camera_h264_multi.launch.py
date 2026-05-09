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
        'v4l2_camera_h264_multi.yaml')
    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to V4L2 + H264 + RTSP YAML parameters file.'),
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
            if 'transform_compute_mode' in camera:
                params['transform_compute_mode'] = camera['transform_compute_mode']

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
                'codec': encoder.get('codec', 0),
                'input_width': encoder.get('input_width', 1920),
                'input_height': encoder.get('input_height', 1200),
                'qp': encoder.get('qp', 20),
                'rate_control_mode': encoder.get('rate_control_mode', 2),
                'bitrate': encoder.get('bitrate', 12000000),
                'framerate': encoder.get('framerate', 30),
                'hw_preset_type': encoder.get('hw_preset_type', 0),
                'profile': encoder.get('profile', 0),
                'iframe_interval': encoder.get('iframe_interval', 5),
                'config': encoder.get('config', 'pframe_cqp'),
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

        streams = params_data.get('streams', [])
        if streams and not isinstance(streams, list):
            raise RuntimeError('YAML "streams" must be a list')

        rtsp_nodes = []
        for index, stream in enumerate(streams or []):
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

            rtsp_nodes.append(
                ComposableNode(
                    name=name,
                    namespace=namespace,
                    package='isaac_ros_rtsp_server',
                    plugin='isaac_ros_rtsp_server::RtspServerNode',
                    parameters=[params],
                    extra_arguments=extra_arguments,
                )
            )

        container = ComposableNodeContainer(
            name='v4l2_h264_rtsp_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=camera_nodes + encoder_nodes + rtsp_nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(
        function=launch_setup)])

