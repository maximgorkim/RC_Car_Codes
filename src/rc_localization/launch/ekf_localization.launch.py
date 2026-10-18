"""Low-latency localization: twist_estimator + ekf_localizer (+ optional monitor).

  ros2 launch rc_localization ekf_localization.launch.py
  ros2 launch rc_localization ekf_localization.launch.py use_sim_time:=true monitor:=true   # with ros2 bag play --clock

The particle filter must keep running (it provides the LiDAR pose). Only one
node may publish the odom -> base_link TF: run the particle filter without its
TF, or start this with publish_tf:=false for a side-by-side comparison.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('rc_localization')
    use_sim_time = LaunchConfiguration('use_sim_time')
    publish_tf = LaunchConfiguration('publish_tf')
    pose_stamp_offset = LaunchConfiguration('pose_stamp_offset')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('publish_tf', default_value='true',
                              description='publish odom->base_link (turn the particle filter TF off)'),
        DeclareLaunchArgument('pose_stamp_offset', default_value='0.20',
                              description='s; LiDAR pose latency not reflected in its stamp'),
        DeclareLaunchArgument('output_trigger', default_value='timer',
                              description="timer (car) | twist (simulation with a coarse /clock)"),
        DeclareLaunchArgument('lateral_dynamics', default_value='false',
                              description='side-slip model (IMU lateral acceleration); true for high speed'),
        DeclareLaunchArgument('accel_scale', default_value='1.24',
                              description='IMU lateral accel scale: 1.24 car (calibrated), 1.0 Gazebo'),
        DeclareLaunchArgument('ay_bias', default_value='-0.30',
                              description='initial IMU lateral accel bias [m/s^2]: -0.30 car, 0.0 Gazebo'),
        DeclareLaunchArgument('monitor', default_value='false',
                              description='also start localization_monitor'),
        DeclareLaunchArgument('csv_path', default_value='',
                              description='monitor CSV log file (empty: no log)'),

        Node(
            package='rc_localization',
            executable='twist_estimator',
            name='twist_estimator',
            output='screen',
            parameters=[os.path.join(share, 'config', 'twist_estimator.yaml'),
                        {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rc_localization',
            executable='ekf_localizer',
            name='ekf_localizer',
            output='screen',
            parameters=[os.path.join(share, 'config', 'ekf_localizer.yaml'),
                        {'use_sim_time': use_sim_time,
                         'publish_tf': publish_tf,
                         'pose_stamp_offset': pose_stamp_offset,
                         'output_trigger': LaunchConfiguration('output_trigger'),
                         'model.lateral_dynamics': LaunchConfiguration('lateral_dynamics'),
                         'accel_scale': LaunchConfiguration('accel_scale'),
                         'init.ay_bias': LaunchConfiguration('ay_bias')}],
        ),
        Node(
            package='rc_localization',
            executable='localization_monitor.py',
            name='localization_monitor',
            output='screen',
            condition=IfCondition(LaunchConfiguration('monitor')),
            parameters=[{'use_sim_time': use_sim_time,
                         'pose_stamp_offset': pose_stamp_offset,
                         'csv_path': LaunchConfiguration('csv_path')}],
        ),
    ])
