from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node

PARAMS = str(Path.home() / 'greenhouse_ws' / 'src' / 'greenhouse_sim' / 'config' / 'nav2_dwb.yaml')


def generate_launch_description():
    common = {'use_sim_time': True}

    return LaunchDescription([
        Node(package='nav2_controller', executable='controller_server',
             output='screen', parameters=[PARAMS, common]),

        Node(package='nav2_planner', executable='planner_server',
             output='screen', parameters=[PARAMS, common]),

        Node(package='nav2_behaviors', executable='behavior_server',
             output='screen', parameters=[PARAMS, common]),

        Node(package='nav2_bt_navigator', executable='bt_navigator',
             output='screen', parameters=[PARAMS, common]),

        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[PARAMS, common]),
    ])
