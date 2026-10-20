from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    ydlidar_dir = get_package_share_directory('ydlidar_ros2_driver')
    vesc_dir = get_package_share_directory('vesc_driver')

    return LaunchDescription([

        # lidar_params:=ydlidar_single_channel -> old config (lidar stuck at ~7 Hz)
        DeclareLaunchArgument('lidar_params', default_value='ydlidar',
                              description='ydlidar_ros2_driver/params/<name>.yaml'),

        # -------------------------
        # YDLIDAR LAUNCH
        # -------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ydlidar_dir, 'launch', 'ydlidar_launch.py')
            ),
            launch_arguments={'params_file': [os.path.join(ydlidar_dir, 'params', ''),
                                              LaunchConfiguration('lidar_params'), '.yaml']}.items()
        ),

        # -------------------------
        # VESC DRIVER LAUNCH
        # -------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(vesc_dir, 'launch', 'vesc_driver_node.launch.py')
            )
        ),

        # -------------------------
        # JOYSTICK NODE (ORTAK)
        # -------------------------
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),
    ])
