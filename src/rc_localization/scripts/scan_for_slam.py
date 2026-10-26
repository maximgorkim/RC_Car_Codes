#!/usr/bin/env python3
"""/scan -> /scan_slam with invalid beams as +inf, for slam_toolbox.

The YDLidar driver (invalid_range_is_inf: false) publishes invalid beams as 0.0
(23-30 % of the beams on the car's bags). slam_toolbox takes its minimum range
from the message's range_min (0.01), not from min_laser_range, and Karto keeps
the 0.0 beams as points at the sensor in parts of the scan matching: on a
synthetic bag with ground truth the map fell apart (mean error 6.5 m, 1 m/s);
with the zeros as +inf it is 29 cm (odometry alone 52 cm).

stamp_shift [s] (default 0) moves the stamp: the driver stamps the scan's first
beam while the beams spread over the scan (0.1 s at 10 Hz); slam_toolbox takes the
stamp as the time of the whole scan, so at 2.5 m/s its pose falls ~12 cm behind
along the track (synthetic bag). scan_time / 2 puts the stamp mid-scan.

The particle filter keeps the raw /scan (range_libc does not take +inf), so the
driver setting stays as it is and only slam_toolbox gets this topic.

  ros2 run rc_localization scan_for_slam.py
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

try:  # Humble: raised by spin() on Ctrl+C / SIGTERM; does not exist on Foxy
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover
    ExternalShutdownException = KeyboardInterrupt


class ScanForSlam(Node):

    def __init__(self):
        super().__init__('scan_for_slam')
        in_topic = self.declare_parameter('in_topic', '/scan').value
        out_topic = self.declare_parameter('out_topic', '/scan_slam').value
        self.shift_ns = int(round(1e9 * self.declare_parameter('stamp_shift', 0.0).value))
        self.pub = self.create_publisher(LaserScan, out_topic, qos_profile_sensor_data)
        self.create_subscription(LaserScan, in_topic, self.on_scan, qos_profile_sensor_data)
        self.get_logger().info(f'{in_topic} -> {out_topic} (beams below range_min as +inf)')

    def on_scan(self, msg):
        lo = msg.range_min
        msg.ranges = [r if r >= lo else math.inf for r in msg.ranges]  # NaN fails r >= lo too
        if self.shift_ns:
            ns = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec + self.shift_ns
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1000000000)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ScanForSlam()
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
