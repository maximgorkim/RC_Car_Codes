# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PythonExpression
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    # config and args
    localize_config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'localize.yaml'
    )
    localize_config_dict = yaml.safe_load(open(localize_config, 'r'))
    map_name = localize_config_dict['map_server']['ros__parameters']['map']
    localize_la = DeclareLaunchArgument(
        'localize_config',
        default_value=localize_config,
        description='Localization configs')
    # ekf:=true -> config/localize_ekf.yaml: no TF, one update per scan stamped with
    # the scan time, motion from the VESC + IMU twist (pose source of rc_localization)
    ekf_la = DeclareLaunchArgument('ekf', default_value='false',
                                   description='true: localize_ekf.yaml (PF as the EKF pose source)')
    ekf_config = os.path.join(get_package_share_directory('particle_filter'), 'config', 'localize_ekf.yaml')
    config = PythonExpression(["'", ekf_config, "' if '", LaunchConfiguration('ekf'), "' == 'true' else '",
                               LaunchConfiguration('localize_config'), "'"])
    # scan_sync:=true -> one update per scan, motion up to the scan's measurement
    # time and the pose stamped with it (always on with ekf:=true). Consumers must
    # then use the stamp: EKF pose_stamp_offset:=0.0, MPC pf_scan_sync:=true
    scan_sync_la = DeclareLaunchArgument('scan_sync', default_value='false',
                                         description='true: pose per scan, stamped with the scan time')
    scan_sync = PythonExpression(["'", LaunchConfiguration('ekf'), "' == 'true' or '",
                                  LaunchConfiguration('scan_sync'), "' == 'true'"])
    # motion_noise:=proportional -> motion noise in the car frame, proportional to
    # the motion since the last update (default 'fixed': the config dispersions)
    motion_noise_la = DeclareLaunchArgument('motion_noise', default_value='fixed',
                                            description='fixed | proportional')
    # twist:=true -> motion model from the VESC + IMU twist instead of ZED VO (always on with ekf:=true)
    twist_la = DeclareLaunchArgument('twist', default_value='false',
                                     description='true: motion from /localization/twist_with_cov (needs twist_estimator)')
    motion_source = PythonExpression(["'twist' if '", LaunchConfiguration('ekf'), "' == 'true' or '",
                                      LaunchConfiguration('twist'), "' == 'true' else 'odom'"])
    # deskew:=true -> every LiDAR beam moved to the pose at the scan's measurement time (scan_sync only).
    # Synthetic bag with ground truth (tools/synth), 2.5 m/s: 3.7 -> 1.6 cm, yaw 1.1 -> 0.2 deg with
    # twist:=true. NOT yet checked on the car: run tools/deskew_check.py on a new bag first.
    deskew_la = DeclareLaunchArgument('deskew', default_value='false',
                                      description='true: deskew the scan with the motion source (scan_sync only)')
    ld = LaunchDescription([localize_la, ekf_la, scan_sync_la, motion_noise_la, twist_la, deskew_la])

    # nodes
    pf_node = Node(
        package='particle_filter',
        executable='particle_filter',
        name='particle_filter',
        parameters=[config, {'scan_synchronous': scan_sync,
                             'motion_noise': LaunchConfiguration('motion_noise'),
                             'motion_source': motion_source,
                             'scan_deskew': PythonExpression(["'", LaunchConfiguration('deskew'), "' == 'true'"])}]
    )
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': os.path.join(get_package_share_directory('particle_filter'), 'maps', map_name + '.yaml')},
                    {'topic': 'map'},
                    {'frame_id': 'odom'},
                    {'output': 'screen'},
                    {'use_sim_time': False}]
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': False},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )

    # finalize
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    ld.add_action(pf_node)

    return ld
