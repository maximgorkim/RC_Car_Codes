"""MPC path-tracking controller.

    # real car (Jetson): drivers + localisation + base stack, then
    ros2 launch mpc_tracking_controller mpc.launch.py
    # Gazebo car-like mode (RC_Car_Sim vehicle:=car): sim time, no arming, the VESC bridge
    ros2 launch mpc_tracking_controller mpc.launch.py use_sim_time:=true arming:=false

localization:=pf (default) tracks the particle filter's /pf/pose/odom and
compensates its latency itself (model.pose_latency_*). localization:=ekf
tracks rc_localization's /localization/odom, which is already predicted to
the present, so that compensation is switched off (it would be applied twice).
The command dead times / actuator prediction stay on in both modes.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    sim = LaunchConfiguration("use_sim_time").perform(context) == "true"
    arming = LaunchConfiguration("arming").perform(context) == "true"
    cfg = LaunchConfiguration("config").perform(context)
    over = {"use_sim_time": sim, "require_arming": arming}
    localization = LaunchConfiguration("localization").perform(context)
    if localization == "ekf":
        over.update({"odom_topic": "/localization/odom",
                     "model.pose_latency_position": 0.0,
                     "model.pose_latency_heading": 0.0})
    elif localization != "pf":
        raise RuntimeError(f"localization:={localization}: expected 'pf' or 'ekf'")
    for name in ("v_max", "a_lat_max"):  # optional overrides of the config's speed limits
        value = LaunchConfiguration(name).perform(context)
        if value:
            over[name] = float(value)
    path_topic = LaunchConfiguration("path_topic").perform(context)
    if path_topic:
        over["path_topic"] = path_topic
    actions = [Node(package="mpc_tracking_controller", executable="mpc_node", name="mpc_controller",
                    output="screen", parameters=[cfg, over])]
    if sim:
        # Gazebo publishes ERPM as Float64; the car's VESC publishes VescStateStamped on /sensors/core
        actions.append(Node(package="rc_localization", executable="sim_vesc_bridge.py", name="sim_vesc_bridge",
                            output="screen", parameters=[{"use_sim_time": True}]))
    return actions


def generate_launch_description():
    share = get_package_share_directory("mpc_tracking_controller")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("arming", default_value="true", description="joystick arming (false in simulation)"),
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "mpc_car.yaml")),
        DeclareLaunchArgument("path_topic", default_value="", description="override, e.g. /selected_path"),
        DeclareLaunchArgument("localization", default_value="pf", description="pf | ekf (see above)"),
        DeclareLaunchArgument("v_max", default_value="", description="m/s, override the config's speed limit"),
        DeclareLaunchArgument("a_lat_max", default_value="",
                              description="m/s^2, override the curve speed cap sqrt(a_lat_max/|kappa|)"),
        OpaqueFunction(function=setup),
    ])
