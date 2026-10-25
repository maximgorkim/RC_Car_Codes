"""Mapping with slam_toolbox: EKF dead reckoning as odometry + scan_for_slam + slam_toolbox (sync).

  ros2 launch rc_localization mapping.launch.py
  ros2 launch rc_localization mapping.launch.py use_sim_time:=true   # start ros2 bag play --clock FIRST

Needs the drivers (LiDAR, VESC) and the ZED IMU. The particle filter must NOT run
(it publishes odom -> base_link); here the EKF does, from IMU + VESC only, and
slam_toolbox publishes map -> odom. Keep the car still for the first seconds: the
EKF is started at the origin by a single /initialpose. Drive slowly (~1 m/s) and
close the loops; save the map with the slam_toolbox services (see the README).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('rc_localization')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('slam_params', default_value=os.path.join(share, 'config', 'slam_mapping.yaml')),
        DeclareLaunchArgument('init_delay', default_value='2.0',
                              description='s until the EKF is started at the origin (bag: after /clock runs)'),

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
                         'publish_tf': True,
                         'pose_topic': '/mapping/no_pose',   # no LiDAR pose: dead reckoning
                         'max_position_variance': 1.0e9}],    # never "diverged" without poses
        ),
        # The EKF waits for a LiDAR pose or an initial pose: start it at the origin.
        TimerAction(period=LaunchConfiguration('init_delay'), actions=[ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '--once', '/initialpose',
                 'geometry_msgs/msg/PoseWithCovarianceStamped', '{header: {frame_id: odom}}'],
            output='log')]),
        Node(
            package='rc_localization',
            executable='scan_for_slam.py',
            name='scan_for_slam',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='slam_toolbox',
            executable='sync_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[LaunchConfiguration('slam_params'), {'use_sim_time': use_sim_time}],
        ),
    ])
