"""Everything on the host side of a Gazebo test (RC_Car_Sim, vehicle:=car):
path + speed reference, EKF, MPC (+ sim_vesc_bridge), all on sim time.

  ros2 launch rc_localization sim_test.launch.py                          # EKF + MPC, stack speed reference
  ros2 launch rc_localization sim_test.launch.py localization:=pf         # old: particle filter pose + MPC
  ros2 launch rc_localization sim_test.launch.py target_speed:=3.0        # constant speed instead
                                                    (v_max follows it; curves stay capped by a_lat_max)
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    vref_cfg = os.path.join(get_package_share_directory('v_ref_gen'), 'config')
    # v_ref_gen's path_file is relative to the workspace root
    workspace = os.path.dirname(os.path.dirname(get_package_prefix('v_ref_gen')))
    sim = {'use_sim_time': True}
    target = LaunchConfiguration('target_speed').perform(context)
    v_max = LaunchConfiguration('v_max').perform(context) or target  # else the MPC config's 2.0 caps it

    actions = [
        Node(package='v_ref_gen', executable='v_ref_gen', name='v_ref_gen', output='screen', cwd=workspace,
             parameters=[os.path.join(vref_cfg, 'path_params.yaml'), sim]),
    ]
    if target:
        actions.append(ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '-r', '20', '/target_speed', 'std_msgs/msg/Float64', f'{{data: {target}}}'],
            output='log'))
    else:
        actions.append(Node(package='v_ref_gen', executable='desired_speed_pub', name='desired_speed_pub',
                            output='screen', parameters=[os.path.join(vref_cfg, 'params.yaml'), sim]))

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rc_localization'), 'launch', 'ekf_localization.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'publish_tf': 'false', 'monitor': 'true',
                          'accel_scale': '1.0', 'ay_bias': '0.0',        # Gazebo IMU is ideal
                          'lateral_dynamics': LaunchConfiguration('lateral_dynamics')}.items()))
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('mpc_tracking_controller'), 'launch', 'mpc.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'arming': 'false', 'path_topic': '/path',
                          'localization': LaunchConfiguration('localization'),
                          'v_max': v_max, 'a_lat_max': LaunchConfiguration('a_lat_max')}.items()))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('localization', default_value='ekf', description='ekf | pf (pose used by the MPC)'),
        DeclareLaunchArgument('lateral_dynamics', default_value='true', description='EKF side-slip model'),
        DeclareLaunchArgument('target_speed', default_value='',
                              description='m/s constant speed reference; empty: desired_speed_pub'),
        DeclareLaunchArgument('v_max', default_value='', description='m/s MPC speed limit; empty: target_speed or config'),
        DeclareLaunchArgument('a_lat_max', default_value='', description='m/s^2 MPC curve cap; empty: config (2.0)'),
        OpaqueFunction(function=setup),
    ])
