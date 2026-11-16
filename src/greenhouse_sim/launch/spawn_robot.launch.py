import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MODEL = '/opt/ros/jazzy/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf'

def launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)
    gui = LaunchConfiguration('gui').perform(context).lower() in ('true', '1', 'yes')
    meta = json.loads(Path(world).with_suffix('.json').read_text())
    start = meta['start_pose']

    print(f"[greenhouse] world: {meta['world_name']}")
    print(f"[greenhouse] start: x={start['x']} y={start['y']} yaw={start['yaw']}")

    gz_cmd = ['gz', 'sim', '-v', '3', '-r']
    if not gui:
        gz_cmd.append('-s')
    gz_cmd.append(world)

    gz = ExecuteProcess(cmd=gz_cmd, output='screen')

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-file', MODEL,
            '-name', 'waffle',
            '-x', str(start['x']),
            '-y', str(start['y']),
            '-z', '0.01',
            '-Y', str(start['yaw']),
        ],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file': str(Path.home() / 'greenhouse_ws/src/greenhouse_sim/config/bridge.yaml'),
            'use_sim_time': True,
        }],
    )

    truth = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[f"/world/{meta['world_name']}/dynamic_pose/info"
                   f"@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
        remappings=[(f"/world/{meta['world_name']}/dynamic_pose/info",
                     '/truth_tf')],
        parameters=[{'use_sim_time': True}],
    )

    urdf = Path('/opt/ros/jazzy/share/turtlebot3_gazebo/urdf/turtlebot3_waffle.urdf').read_text()

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': urdf,
            'use_sim_time': True,
        }],
    )

    return [gz, TimerAction(period=4.0,
                            actions=[spawn, bridge, truth, rsp])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', description='Path to a generated .sdf world'),
        DeclareLaunchArgument('gui', default_value='true'),
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            '/opt/ros/jazzy/share/turtlebot3_gazebo/models'
        ),
        OpaqueFunction(function=launch_setup),
    ])
