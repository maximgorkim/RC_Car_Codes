#!/usr/bin/env python3
"""Simulation only: RC_Car_Sim car-like mode -> the car's VESC message.

RC_Car_Sim (vehicle:=car) publishes the VESC's ERPM as std_msgs/Float64 on
/sim/vesc/erpm; the car's vesc_driver publishes vesc_msgs/VescStateStamped on
/sensors/core (state.speed = ERPM). This republishes the former as the latter,
stamped on reception like vesc_driver does, so twist_estimator and the MPC run
unchanged in simulation.

  ros2 run rc_localization sim_vesc_bridge.py --ros-args -p use_sim_time:=true
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped

try:  # Humble: raised by spin() on Ctrl+C / SIGTERM; does not exist on Foxy
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover
    ExternalShutdownException = KeyboardInterrupt


class SimVescBridge(Node):

    def __init__(self):
        super().__init__('sim_vesc_bridge')
        in_topic = self.declare_parameter('erpm_topic', '/sim/vesc/erpm').value
        out_topic = self.declare_parameter('vesc_state_topic', '/sensors/core').value
        self.pub = self.create_publisher(VescStateStamped, out_topic, 10)
        self.create_subscription(Float64, in_topic, self.on_erpm, 10)
        self.get_logger().info(f'{in_topic} (Float64 ERPM) -> {out_topic} (VescStateStamped)')

    def on_erpm(self, msg):
        out = VescStateStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.state.speed = msg.data
        self.pub.publish(out)


def main():
    rclpy.init()
    node = SimVescBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
