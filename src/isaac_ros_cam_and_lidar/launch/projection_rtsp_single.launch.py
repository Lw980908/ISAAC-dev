import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('isaac_ros_cam_and_lidar'),
        'config',
        'projection_rtsp_single.yaml')

    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to single-camera projection RTSP YAML parameters file.'),
    ]

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)
        with open(params_path, 'r', encoding='utf-8') as handle:
            params_data = yaml.safe_load(handle) or {}

        cameras = params_data.get('cameras', [])
        if not isinstance(cameras, list) or not cameras:
          raise RuntimeError('YAML "cameras" must be a non-empty list')

        encoders = params_data.get('encoders', [])
        if not isinstance(encoders, list) or not encoders:
          raise RuntimeError('YAML "encoders" must be a non-empty list')

        streams = params_data.get('streams', [])
        if not isinstance(streams, list) or not streams:
          raise RuntimeError('YAML "streams" must be a non-empty list')

        projection_demo_cfg = params_data.get('projection_demo', {})
        if not isinstance(projection_demo_cfg, dict):
          raise RuntimeError('YAML "projection_demo" must be a map')

        enable_point_cloud_rendering = params_data.get('enable_point_cloud_rendering', True)
        point_cloud_renderer_cfg = params_data.get('point_cloud_renderer', {})
        if not isinstance(point_cloud_renderer_cfg, dict):
          raise RuntimeError('YAML "point_cloud_renderer" must be a map')
        enable_point_cloud_renderer = (
            enable_point_cloud_rendering and point_cloud_renderer_cfg.get('enable', True))
        point_cloud_renderer_params = {
            key: value
            for key, value in point_cloud_renderer_cfg.items()
            if key != 'enable'
        }

        camera_nodes = []
        for index, camera in enumerate(cameras):
            if not isinstance(camera, dict):
                raise RuntimeError(f'camera[{index}] must be a map')
            name = camera.get('name', f'v4l2_gpu_camera_{index}')
            namespace = camera.get('namespace', '')

            params = {
                'device': camera.get('device', f'/dev/video{index}'),
                'camera_info_url': camera.get('camera_info_url', ''),
                'camera_link_frame_name': camera.get(
                    'camera_link_frame_name', f'camera_{index}'),
                'optical_frame_name': camera.get(
                    'optical_frame_name', f'camera_{index}_optical'),
                'image_topic': camera.get('image_topic', 'image_raw'),
                'camera_info_topic': camera.get('camera_info_topic', 'camera_info'),
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
                    namespace=namespace,
                    package='isaac_ros_v4l2_camera',
                    plugin='nvidia::isaac_ros::v4l2_camera::V4l2GpuCameraNode',
                    parameters=[params],
                )
            )

        encoder_nodes = []
        for index, encoder in enumerate(encoders):
            if not isinstance(encoder, dict):
                raise RuntimeError(f'encoder[{index}] must be a map')

            params = {
                'codec': encoder.get('codec', 0),
                'input_width': encoder.get('input_width', 1920),
                'input_height': encoder.get('input_height', 1536),
                'qp': encoder.get('qp', 20),
                'rate_control_mode': encoder.get('rate_control_mode', 2),
                'bitrate': encoder.get('bitrate', 12000000),
                'framerate': encoder.get('framerate', 30),
                'hw_preset_type': encoder.get('hw_preset_type', 0),
                'profile': encoder.get('profile', 0),
                'iframe_interval': encoder.get('iframe_interval', 5),
                'config': encoder.get('config', 'pframe_cqp'),
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
                    name=encoder.get('name', f'h264_encoder_{index}'),
                    namespace=encoder.get('namespace', ''),
                    package='isaac_ros_h264_encoder',
                    plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
                    parameters=[params],
                    remappings=remappings,
                )
            )

        rtsp_nodes = []
        for index, stream in enumerate(streams):
            if not isinstance(stream, dict):
                raise RuntimeError(f'stream[{index}] must be a map')
            if stream.get('enable', True) is not True:
                continue

            params = {
                'stream_name': stream.get('stream_name', f'projection_{index}'),
                'port': stream.get('port', 8554),
                'video_type': stream.get('video_type', 'h264'),
                'topic_name': stream.get('topic_name', 'image_compressed'),
                'assume_annexb': stream.get('assume_annexb', True),
                'dump_frame': stream.get('dump_frame', 0),
                'use_nitros': stream.get('use_nitros', True),
                'nitros_format': stream.get('nitros_format', 'nitros_compressed_image'),
                'max_queue_frames': stream.get('max_queue_frames', 2),
            }

            rtsp_nodes.append(
                ComposableNode(
                    name=stream.get('name', f'rtsp_server_{index}'),
                    namespace=stream.get('namespace', ''),
                    package='isaac_ros_rtsp_server',
                    plugin='isaac_ros_rtsp_server::RtspServerNode',
                    parameters=[params],
                )
            )

        projection_demo_node = ComposableNode(
            name='camera_lidar_projection_demo',
            namespace='',
            package='isaac_ros_cam_and_lidar',
            plugin='CameraLidarProjectionDemo',
            parameters=[projection_demo_cfg],
        )

        point_cloud_renderer_nodes = []
        if enable_point_cloud_renderer:
            point_cloud_renderer_nodes.append(
                ComposableNode(
                    name='colored_cloud_view_renderer',
                    namespace='',
                    package='isaac_ros_cam_and_lidar',
                    plugin='ColoredCloudViewRenderer',
                    parameters=[point_cloud_renderer_params],
                )
            )

        container = ComposableNodeContainer(
            name='projection_rtsp_single_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=(
                camera_nodes + [projection_demo_node] + point_cloud_renderer_nodes +
                encoder_nodes + rtsp_nodes),
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
