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
启动文件：MP4 多视频发布(NITROS NV12) -> 感知(可选) -> H264 编码器 -> RTSP 推流(可选)

数据流：
  Mp4Nv12NitrosPublisherNode -> NitrosImage (NV12) -> PerceptionComponent -> NitrosImage (NV12) -> H264 Encoder -> RTSP
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
        'v4l2_video_perception_h264.yaml')

    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to combined MP4+Perception+H264 YAML parameters file.'),
        DeclareLaunchArgument(
            'enable_perception',
            default_value='true',
            description='Enable perception node (object detection).'),
    ]

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)
        enable_perception = LaunchConfiguration(
            'enable_perception').perform(context).lower() == 'true'

        with open(params_path, 'r', encoding='utf-8') as handle:
            params_data = yaml.safe_load(handle) or {}

        # ==================== 1. 视频发布节点（多路）====================
        videos = params_data.get('videos', [])
        if not isinstance(videos, list):
            raise RuntimeError('YAML "videos" must be a list')

        video_nodes = []
        video_namespaces = []
        input_sizes_from_videos = []
        camera_ids_from_videos = []

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
                    f'video[{index}] must set width/height (>0) for downstream perception/encoder')

            video_namespaces.append(namespace)
            input_sizes_from_videos.extend([w, h])
            camera_ids_from_videos.append(index)

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

        if not video_nodes:
            raise RuntimeError('No videos configured in YAML')

        # ==================== 2. 感知节点（可选）====================
        perception_nodes = []
        if enable_perception:
            perception_config = params_data.get('perception', {})

            process_method_value = perception_config.get(
                'process_method', 'SHARED_MODEL_THREAD_MULTI_SOURCES')
            process_method_map = {
                'MULTI_THREADS_MULTI_SOURCES': 0,
                'SINGLE_THREAD_SINGLE_SOURCE': 1,
                'SHARED_MODEL_THREAD_MULTI_SOURCES': 2,
            }
            if isinstance(process_method_value, str):
                process_method_int = process_method_map.get(
                    process_method_value, process_method_map['SHARED_MODEL_THREAD_MULTI_SOURCES'])
            elif isinstance(process_method_value, int):
                process_method_int = int(process_method_value)
            else:
                process_method_int = process_method_map['SHARED_MODEL_THREAD_MULTI_SOURCES']

            num_sources = perception_config.get('num_sources', len(videos))
            perception_video_namespaces = perception_config.get(
                'camera_namespaces', video_namespaces)
            camera_ids = perception_config.get('camera_ids', camera_ids_from_videos)
            input_sizes = perception_config.get('input_sizes', input_sizes_from_videos)

            default_roi_regions = []
            for i, video in enumerate(videos):
                w = int(video.get('width', 0))
                h = int(video.get('height', 0))
                default_roi_regions.extend([0, 0, w, h])
            roi_regions = perception_config.get('roi_regions', default_roi_regions)

            perception_params = {
                'num_sources': num_sources,
                'camera_namespaces': perception_video_namespaces,
                'camera_ids': camera_ids,
                'input_sizes': input_sizes,
                'roi_regions': roi_regions,
                'process_method': process_method_int,
                'ros2_enabled': perception_config.get('ros2_enabled', True),
                'use_ros2_component_container': perception_config.get('use_ros2_component_container', True),
                'encoding_format_sub': perception_config.get('encoding_format_sub', 'NITROS_NV12'),
                'encoding_format_pub': perception_config.get('encoding_format_pub', 'NITROS_NV12'),
                'nitros_topic_prefix': perception_config.get('nitros_topic_prefix', ''),
                'nitros_pub_topic_prefix': perception_config.get('nitros_pub_topic_prefix', ''),
                'show_gui': perception_config.get('show_gui', False),
            }

            perception_nodes.append(
                ComposableNode(
                    name='perception_node',
                    package='object_detection',
                    plugin='object_detection::PerceptionComponent',
                    namespace='',
                    parameters=[perception_params],
                )
            )

        # ==================== 3. H264 编码器节点 ====================
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
            }

            if enable_perception:
                input_topic = encoder.get('input_topic_with_perception', 'processed_image')
            else:
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

        # ==================== 4. RTSP 推流节点（可选）====================
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
                'max_queue_frames': stream.get('max_queue_frames', 128),
                'use_nitros': use_nitros,
                'nitros_format': stream.get('nitros_format', 'nitros_compressed_image'),
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

        # ==================== 5. 组合所有节点到容器 ====================
        all_nodes = video_nodes + perception_nodes + encoder_nodes + rtsp_nodes

        container = ComposableNodeContainer(
            name='video_perception_h264_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=all_nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
