from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
import os
from launch.actions import SetLaunchConfiguration, LogInfo, GroupAction
from launch.substitutions import PythonExpression


def generate_launch_description():
    # 调试模式参数声明
    use_gdb_arg = DeclareLaunchArgument(
        'use_gdb',
        default_value='true',
        description='Enable GDB debugging in current terminal'
    )

    gdb_commands_arg = DeclareLaunchArgument(
        'gdb_commands',
        default_value='break main break SharedModelMultiSourcesPipelineManager::start break SharedModelMultiSourcesPipelineManager::processDetectionGlobal',
        description='GDB breakpoints to set'
    )

    # 根据条件创建不同的节点配置
    debug_node_with_gdb = Node(
        package='object_detection',
        executable='object_detection_node',
        output='screen',
        emulate_tty=True,
        # 直接使用GDB前缀，不通过PythonExpression
        prefix=['gdb -ex "',
                LaunchConfiguration('gdb_commands'), '" -ex run --args '],
        parameters=[{
            'use_sim_time': False,
            'debug_mode': LaunchConfiguration('use_gdb'),
        }],
        additional_env={
            'GDB_DEBUG': LaunchConfiguration('use_gdb'),
            'ROS_LOG_DIR': '/tmp/ros2_logs'
        },
        condition=IfCondition(LaunchConfiguration(
            'use_gdb'))  # 仅当use_gdb为true时执行
    )

    debug_node_normal = Node(
        package='object_detection',
        executable='object_detection_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': False,
            'debug_mode': LaunchConfiguration('use_gdb'),
        }],
        additional_env={
            'GDB_DEBUG': LaunchConfiguration('use_gdb'),
            'ROS_LOG_DIR': '/tmp/ros2_logs'
        },
        condition=UnlessCondition(LaunchConfiguration(
            'use_gdb'))  # 仅当use_gdb为false时执行
    )

    # 调试信息输出
    debug_info = LogInfo(
        condition=IfCondition(LaunchConfiguration('use_gdb')),
        msg=['启动GDB调试模式，断点设置: ', LaunchConfiguration('gdb_commands')]
    )

    normal_info = LogInfo(
        condition=UnlessCondition(LaunchConfiguration('use_gdb')),
        msg=['启动正常模式（无调试器）']
    )

    # 进程启动事件处理
    def on_process_start(event, context):
        return [LogInfo(msg=f"进程已启动，PID: {event.pid}")]

    # 进程退出事件处理
    def on_process_exit(event, context):
        exit_code = event.returncode
        if exit_code == 0:
            msg = f"进程正常退出，退出码: {exit_code}"
        elif exit_code == -11:  # Segmentation fault
            msg = f"进程段错误退出(Segmentation Fault)，退出码: {exit_code}"
        else:
            msg = f"进程异常退出，退出码: {exit_code}"

        debug_advice = ""
        if exit_code == -11:
            debug_advice = "\n建议：检查多线程同步和内存访问问题，使用GDB分析core dump"

        return [
            ExecuteProcess(
                cmd=['echo', f'{msg}{debug_advice}'],
                shell=True,
                output='screen'
            ),
            LogInfo(msg=f"{msg}{debug_advice}")
        ]

    return LaunchDescription([
        use_gdb_arg,
        gdb_commands_arg,
        debug_info,
        normal_info,
        debug_node_with_gdb,  # 调试模式节点
        debug_node_normal,    # 正常模式节点

        # 进程启动事件
        RegisterEventHandler(
            OnProcessStart(
                target_action=debug_node_with_gdb,  # 监控调试节点
                on_start=on_process_start
            )
        ),

        RegisterEventHandler(
            OnProcessStart(
                target_action=debug_node_normal,  # 监控正常节点
                on_start=on_process_start
            )
        ),

        # 进程退出事件
        RegisterEventHandler(
            OnProcessExit(
                target_action=debug_node_with_gdb,  # 监控调试节点退出
                on_exit=on_process_exit
            )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=debug_node_normal,  # 监控正常节点退出
                on_exit=on_process_exit
            )
        )
    ])
