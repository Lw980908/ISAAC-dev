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

"""
启动文件：V4L2 相机采集 -> 感知节点（目标检测）-> H264 编码器 -> RTSP 推流

数据流：
  V4l2GpuCameraNode -> NitrosImage (NV12) -> PerceptionComponent -> NitrosImage (NV12) -> H264 Encoder -> RTSP

话题映射：
  - 相机输出: /<namespace>/image_raw (NitrosImage NV12)
  - 感知输入: /<namespace>/image_raw
  - 感知输出: /<namespace>/processed_image (NitrosImage NV12)
  - 编码器输入: /<namespace>/processed_image
  - 编码器输出: /<namespace>/image_compressed (H264)
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
        'v4l2_camera_perception_h264.yaml')

    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to combined V4L2 + Perception + H264 YAML parameters file.'),
        DeclareLaunchArgument(
            'enable_perception',
            default_value='true',
            description='Enable perception node (object detection).'),
        DeclareLaunchArgument(
            'livox_xfer_format',
            default_value='',
            description='Override livox_xfer_format for perception node (0/1). '
                        'Empty uses YAML or node default.'),
    ]

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)
        enable_perception = LaunchConfiguration(
            'enable_perception').perform(context).lower() == 'true'

        with open(params_path, 'r', encoding='utf-8') as handle:
            params_data = yaml.safe_load(handle) or {}

        # ==================== 1. 相机节点 ====================
        cameras = params_data.get('cameras', [])
        if not isinstance(cameras, list):
            raise RuntimeError('YAML "cameras" must be a list')

        camera_nodes = []
        camera_namespaces = []

        for index, camera in enumerate(cameras):
            if not isinstance(camera, dict):
                raise RuntimeError(f'camera[{index}] must be a map')
            name = camera.get('name')
            namespace = camera.get('namespace', '')
            if not name:
                raise RuntimeError(f'camera[{index}] missing "name"')

            camera_namespaces.append(namespace)

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

        # ==================== 2. 感知节点（可选）====================
        perception_nodes = []

        if enable_perception:
            perception_config = params_data.get('perception', {})

            # process_method 支持字符串（推荐）或整数。
            # C++ 侧期望 int，并按 enum class ProcessMethod 的声明顺序：
            #   0: MULTI_THREADS_MULTI_SOURCES
            #   1: SINGLE_THREAD_SINGLE_SOURCE
            #   2: SHARED_MODEL_THREAD_MULTI_SOURCES
            process_method_value = perception_config.get(
                'process_method', 'SHARED_MODEL_THREAD_MULTI_SOURCES')
            process_method_map = {
                'MULTI_THREADS_MULTI_SOURCES': 0,
                'SINGLE_THREAD_SINGLE_SOURCE': 1,
                'SHARED_MODEL_THREAD_MULTI_SOURCES': 2,
                'CAMERA_COMBINED_WITH_LIDAR': 3,
            }
            if isinstance(process_method_value, str):
                process_method_int = process_method_map.get(
                    process_method_value, process_method_map['SHARED_MODEL_THREAD_MULTI_SOURCES'])
            elif isinstance(process_method_value, int):
                process_method_int = int(process_method_value)
            else:
                process_method_int = process_method_map['SHARED_MODEL_THREAD_MULTI_SOURCES']

            # 从相机配置中提取分辨率信息
            # 格式: [width1, height1, width2, height2, ...]
            input_sizes_from_cameras = []
            camera_ids_from_cameras = []
            for idx, camera in enumerate(cameras):
                input_sizes_from_cameras.extend([
                    camera.get('width', 1920),
                    camera.get('height', 1536)
                ])
                camera_ids_from_cameras.append(idx)

            # 优先使用 perception 配置中的值，否则从相机配置提取
            num_sources = perception_config.get('num_sources', len(cameras))

            # camera_namespaces: 优先用 perception 配置，否则从相机配置提取
            perception_camera_namespaces = perception_config.get(
                'camera_namespaces', camera_namespaces)

            # camera_ids: 优先用 perception 配置，否则自动生成
            camera_ids = perception_config.get(
                'camera_ids', camera_ids_from_cameras)

            # input_sizes: 优先用 perception 配置，否则从相机配置提取
            input_sizes = perception_config.get(
                'input_sizes', input_sizes_from_cameras)

            # roi_regions: 使用 perception 配置，或默认为全图
            default_roi_regions = []
            for i in range(len(cameras)):
                w = cameras[i].get('width', 1920)
                h = cameras[i].get('height', 1536)
                default_roi_regions.extend([0, 0, w, h])
            roi_regions = perception_config.get(
                'roi_regions', default_roi_regions)

            # 感知节点参数
            perception_params = {
                # 多路相机配置
                'num_sources': num_sources,
                'camera_namespaces': perception_camera_namespaces,
                'camera_ids': camera_ids,
                'input_sizes': input_sizes,
                'roi_regions': roi_regions,
                # 处理方法
                'process_method': process_method_int,
                # ROS2 配置
                'ros2_enabled': perception_config.get('ros2_enabled', True),
                'use_ros2_component_container': perception_config.get('use_ros2_component_container', True),
                # NITROS 格式配置
                'encoding_format_sub': perception_config.get('encoding_format_sub', 'NITROS_NV12'),
                'encoding_format_pub': perception_config.get('encoding_format_pub', 'NITROS_NV12'),
                # 话题前缀配置
                'nitros_topic_prefix': perception_config.get('nitros_topic_prefix', ''),
                'nitros_pub_topic_prefix': perception_config.get('nitros_pub_topic_prefix', ''),
                # 其他感知参数
                'show_gui': perception_config.get('show_gui', False),
            }

            livox_xfer_format_arg = LaunchConfiguration(
                'livox_xfer_format').perform(context).strip()
            if livox_xfer_format_arg != '':
                perception_params['livox_xfer_format'] = int(livox_xfer_format_arg)
            elif 'livox_xfer_format' in perception_config:
                perception_params['livox_xfer_format'] = int(
                    perception_config.get('livox_xfer_format'))

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

            # 如果启用了感知节点，编码器订阅感知节点的输出
            if enable_perception:
                input_topic = encoder.get(
                    'input_topic_with_perception', 'processed_image')
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

        # ==================== 5. 组合所有节点到容器 ====================
        all_nodes = camera_nodes + perception_nodes + encoder_nodes + rtsp_nodes

        container = ComposableNodeContainer(
            name='v4l2_perception_h264_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=all_nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(
        function=launch_setup)])

