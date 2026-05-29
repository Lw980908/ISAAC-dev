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
        'camera_combined_with_lidar_h264.yaml')

    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to camera + lidar perception + H264 YAML parameters file.'),
        DeclareLaunchArgument(
            'livox_xfer_format',
            default_value='',
            description='Override livox_xfer_format for camera-lidar perception (0/1). '
                        'Empty uses YAML or node default.'),
    ]

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)
        with open(params_path, 'r', encoding='utf-8') as handle:
            params_data = yaml.safe_load(handle) or {}

        cameras = params_data.get('cameras', [])
        if not isinstance(cameras, list) or not cameras:
            raise RuntimeError('YAML "cameras" must be a non-empty list')
        if len(cameras) != 1:
            raise RuntimeError(
                'camera_combined_with_lidar currently supports exactly one camera entry')

        encoders = params_data.get('encoders', [])
        if not isinstance(encoders, list) or not encoders:
            raise RuntimeError('YAML "encoders" must be a non-empty list')

        streams = params_data.get('streams', [])
        if streams and not isinstance(streams, list):
            raise RuntimeError('YAML "streams" must be a list')

        perception_config = params_data.get('perception', {})
        if not isinstance(perception_config, dict):
            raise RuntimeError('YAML "perception" must be a map')

        camera = cameras[0]
        if not isinstance(camera, dict):
            raise RuntimeError('camera[0] must be a map')

        namespace = camera.get('namespace', '')
        if not namespace:
            raise RuntimeError('camera[0] missing "namespace"')

        camera_params = {
            'device': camera.get('device', '/dev/video0'),
            'camera_info_url': camera.get('camera_info_url', ''),
            'camera_link_frame_name': camera.get(
                'camera_link_frame_name', namespace),
            'optical_frame_name': camera.get(
                'optical_frame_name', f'{namespace}_optical'),
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
            camera_params['transform_compute_mode'] = camera['transform_compute_mode']

        camera_node = ComposableNode(
            name=camera.get('name', 'image_acquisition_node'),
            package='isaac_ros_v4l2_camera',
            plugin='nvidia::isaac_ros::v4l2_camera::V4l2GpuCameraNode',
            namespace=namespace,
            parameters=[camera_params],
        )

        camera_id = perception_config.get('camera_ids', [camera.get('camera_id', 0)])
        input_sizes = perception_config.get(
            'input_sizes', [camera_params['width'], camera_params['height']])
        roi_regions = perception_config.get(
            'roi_regions', [0, 0, camera_params['width'], camera_params['height']])

        perception_params = {
            'num_sources': 1,
            'camera_namespaces': perception_config.get('camera_namespaces', [namespace]),
            'camera_ids': camera_id,
            'input_sizes': input_sizes,
            'roi_regions': roi_regions,
            'ros2_enabled': perception_config.get('ros2_enabled', True),
            'use_ros2_component_container': perception_config.get(
                'use_ros2_component_container', True),
            'encoding_format_sub': perception_config.get(
                'encoding_format_sub', 'NITROS_NV12'),
            'encoding_format_pub': perception_config.get(
                'encoding_format_pub', 'NITROS_NV12'),
            'nitros_topic_prefix': perception_config.get('nitros_topic_prefix', ''),
            'nitros_pub_topic_prefix': perception_config.get('nitros_pub_topic_prefix', ''),
            'show_gui': perception_config.get('show_gui', False),
        }

        reserved_keys = {
            'num_sources',
            'camera_namespaces',
            'camera_ids',
            'input_sizes',
            'roi_regions',
            'process_method',
            'ros2_enabled',
            'use_ros2_component_container',
            'encoding_format_sub',
            'encoding_format_pub',
            'nitros_topic_prefix',
            'nitros_pub_topic_prefix',
            'show_gui',
        }
        for key, value in perception_config.items():
            if key not in reserved_keys:
                perception_params[key] = value

        livox_xfer_format_arg = LaunchConfiguration(
            'livox_xfer_format').perform(context).strip()
        if livox_xfer_format_arg != '':
            perception_params['livox_xfer_format'] = int(livox_xfer_format_arg)

        perception_node = ComposableNode(
            name='camera_combined_with_lidar_perception',
            package='isaac_ros_cam_and_lidar',
            plugin='isaac_ros_cam_and_lidar::CameraCombinedWithLidarComponent',
            namespace='',
            parameters=[perception_params],
        )

        encoder_nodes = []
        for index, encoder in enumerate(encoders):
            if not isinstance(encoder, dict):
                raise RuntimeError(f'encoder[{index}] must be a map')

            encoder_params = {
                'codec': encoder.get('codec', 0),
                'input_width': encoder.get('input_width', camera_params['width']),
                'input_height': encoder.get('input_height', camera_params['height']),
                'qp': encoder.get('qp', 20),
                'rate_control_mode': encoder.get('rate_control_mode', 2),
                'bitrate': encoder.get('bitrate', 12000000),
                'framerate': encoder.get('framerate', 30),
                'hw_preset_type': encoder.get('hw_preset_type', 0),
                'profile': encoder.get('profile', 0),
                'iframe_interval': encoder.get('iframe_interval', 5),
                'config': encoder.get('config', 'custom'),
            }
            if 'compatible_input_format' in encoder:
                encoder_params['compatible_input_format'] = encoder['compatible_input_format']

            input_topic = encoder.get('input_topic_with_perception', 'image_perceptual')
            output_topic = encoder.get('output_topic', 'image_compressed')

            remappings = []
            if input_topic != 'image_raw':
                remappings.append(('image_raw', input_topic))
            if output_topic != 'image_compressed':
                remappings.append(('image_compressed', output_topic))

            encoder_nodes.append(
                ComposableNode(
                    name=encoder.get('name', f'encoder_node_{index}'),
                    namespace=encoder.get('namespace', namespace),
                    package='isaac_ros_h264_encoder',
                    plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
                    parameters=[encoder_params],
                    remappings=remappings,
                )
            )

        rtsp_nodes = []
        for index, stream in enumerate(streams or []):
            if not isinstance(stream, dict):
                raise RuntimeError(f'stream[{index}] must be a map')

            use_nitros = stream.get('use_nitros', False)
            rtsp_params = {
                'stream_name': stream.get('stream_name', f'{namespace}_{index}'),
                'port': stream.get('port', 8554),
                'video_type': stream.get('video_type', 'h264'),
                'topic_name': stream.get('topic_name', 'image_compressed'),
                'assume_annexb': stream.get('assume_annexb', True),
                'dump_frame': stream.get('dump_frame', 0),
                'use_nitros': use_nitros,
                'nitros_format': stream.get(
                    'nitros_format', 'nitros_compressed_image'),
            }
            if 'max_queue_frames' in stream:
                rtsp_params['max_queue_frames'] = stream['max_queue_frames']

            extra_arguments = []
            if not use_nitros:
                extra_arguments = [{'use_intra_process_comms': True}]

            rtsp_nodes.append(
                ComposableNode(
                    name=stream.get('name', f'rtsp_stream_node_{index}'),
                    namespace=stream.get('namespace', namespace),
                    package='isaac_ros_rtsp_server',
                    plugin='isaac_ros_rtsp_server::RtspServerNode',
                    parameters=[rtsp_params],
                    extra_arguments=extra_arguments,
                )
            )

        container = ComposableNodeContainer(
            name='camera_combined_with_lidar_h264_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[camera_node, perception_node] + encoder_nodes + rtsp_nodes,
            namespace='',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        return [container]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
