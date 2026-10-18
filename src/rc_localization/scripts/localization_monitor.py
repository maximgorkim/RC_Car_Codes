#!/usr/bin/env python3
"""Localization lag / quality monitor (RC-car version of the big car's ndt_ekf_lag_monitor.py).

Every `report_period` seconds it prints:
  * output rate of the EKF odometry and of the LiDAR pose
  * TF age: now - stamp of the newest world->base transform (target 0-25 ms)
  * EKF vs LiDAR pose: the LiDAR pose is compared with the EKF output at the
    LiDAR measurement time (stamp - pose_stamp_offset), i.e. the residual the
    EKF could not explain
  * jumps: largest change between consecutive EKF outputs beyond what the
    speed explains (target < 10 cm)

With `csv_path` set, every EKF output and LiDAR pose is also logged for plots.

  ros2 run rc_localization localization_monitor.py --ros-args -p use_sim_time:=true
"""

import bisect
import csv
import math
from collections import deque

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

try:  # Humble: raised by spin() on Ctrl+C / SIGTERM; does not exist on Foxy
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover
    ExternalShutdownException = KeyboardInterrupt


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def stamp_sec(stamp):
    return stamp.sec + 1e-9 * stamp.nanosec


class Window:
    """Values received in the current report period."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.values = []

    def add(self, v):
        self.values.append(v)

    def stats(self):
        v = self.values
        if not v:
            return None
        s = sorted(v)
        return {
            'n': len(v),
            'mean': sum(v) / len(v),
            'p95': s[min(len(s) - 1, int(0.95 * len(s)))],
            'max': s[-1],
        }


class LocalizationMonitor(Node):

    def __init__(self):
        super().__init__('localization_monitor')
        self.ekf_topic = self.declare_parameter('ekf_topic', '/localization/odom').value
        self.pose_topic = self.declare_parameter('pose_topic', '/pf/pose/odom').value
        self.pose_stamp_offset = self.declare_parameter('pose_stamp_offset', 0.0).value
        self.world_frame = self.declare_parameter('world_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.report_period = self.declare_parameter('report_period', 2.0).value
        csv_path = self.declare_parameter('csv_path', '').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # EKF output history for lookups at the LiDAR measurement time.
        self.ekf_t = deque(maxlen=500)
        self.ekf_xyz = deque(maxlen=500)
        self.last_ekf = None

        self.n_ekf = 0
        self.n_pose = 0
        self.w_tf_age = Window()
        self.w_out_age = Window()
        self.w_pos_err = Window()
        self.w_yaw_err = Window()
        self.w_jump = Window()
        self.w_pose_age = Window()
        self.total = {'ekf': 0, 'pose': 0, 'max_jump': 0.0, 'pos_err_sq': 0.0, 'n_err': 0}

        self.csv = None
        if csv_path:
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(['source', 't', 'x', 'y', 'yaw', 'v', 'recv_t'])

        self.create_subscription(Odometry, self.ekf_topic, self.on_ekf, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.pose_topic, self.on_pose, qos_profile_sensor_data)
        # 7 ms: not a divisor of the 20 ms output period, so the samples sweep
        # the whole output cycle instead of locking to one phase of it.
        self.create_timer(0.007, self.sample_tf)
        self.create_timer(self.report_period, self.report)
        self.get_logger().info(
            f'monitoring {self.ekf_topic} vs {self.pose_topic} (pose stamp offset {self.pose_stamp_offset:.3f} s), '
            f'TF {self.world_frame}->{self.base_frame}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_ekf(self, msg):
        t = stamp_sec(msg.header.stamp)
        recv = self.now()
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        yaw = yaw_of(msg.pose.pose.orientation)
        v = msg.twist.twist.linear.x
        self.n_ekf += 1
        self.total['ekf'] += 1
        self.w_out_age.add(recv - t)
        if self.last_ekf is not None:
            lt, lx, ly, lv = self.last_ekf
            dt = t - lt
            if dt > 0:
                # Displacement not explained by the speed = jump.
                jump = max(0.0, math.hypot(x - lx, y - ly) - abs(0.5 * (v + lv)) * dt)
                self.w_jump.add(jump)
                self.total['max_jump'] = max(self.total['max_jump'], jump)
        self.last_ekf = (t, x, y, v)
        if self.ekf_t and t <= self.ekf_t[-1]:
            self.ekf_t.clear()
            self.ekf_xyz.clear()
        self.ekf_t.append(t)
        self.ekf_xyz.append((x, y, yaw))
        if self.csv:
            self.csv.writerow(['ekf', f'{t:.6f}', f'{x:.4f}', f'{y:.4f}', f'{yaw:.5f}', f'{v:.3f}', f'{recv:.6f}'])

    def ekf_at(self, t):
        """EKF output linearly interpolated at time t (None if out of range)."""
        ts = self.ekf_t
        if len(ts) < 2 or t < ts[0] or t > ts[-1]:
            return None
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return self.ekf_xyz[0]
        t0, t1 = ts[i - 1], ts[i]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p0, p1 = self.ekf_xyz[i - 1], self.ekf_xyz[i]
        return (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * wrap(p1[2] - p0[2]))

    def on_pose(self, msg):
        recv = self.now()
        t = stamp_sec(msg.header.stamp) - self.pose_stamp_offset
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        yaw = yaw_of(msg.pose.pose.orientation)
        self.n_pose += 1
        self.total['pose'] += 1
        self.w_pose_age.add(recv - t)
        e = self.ekf_at(t)
        if e is not None:
            err = math.hypot(x - e[0], y - e[1])
            self.w_pos_err.add(err)
            self.w_yaw_err.add(abs(wrap(yaw - e[2])))
            self.total['pos_err_sq'] += err * err
            self.total['n_err'] += 1
        if self.csv:
            self.csv.writerow(['pose', f'{t:.6f}', f'{x:.4f}', f'{y:.4f}', f'{yaw:.5f}',
                               f'{msg.twist.twist.linear.x:.3f}', f'{recv:.6f}'])

    def sample_tf(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, Time())
        except Exception:  # noqa: BLE001 - not available yet
            return
        self.w_tf_age.add(self.now() - stamp_sec(tf.header.stamp))

    @staticmethod
    def fmt(w, scale=1.0, unit=''):
        s = w.stats()
        if s is None:
            return 'n/a'
        return f"mean {s['mean'] * scale:6.1f}{unit}  p95 {s['p95'] * scale:6.1f}{unit}  max {s['max'] * scale:6.1f}{unit}"

    def report(self):
        p = self.report_period
        self.get_logger().info(
            f'\n  rate      : EKF {self.n_ekf / p:5.1f} Hz | LiDAR pose {self.n_pose / p:5.1f} Hz'
            f'\n  TF age    : {self.fmt(self.w_tf_age, 1e3, " ms")}'
            f'\n  odom age  : {self.fmt(self.w_out_age, 1e3, " ms")}'
            f'\n  pose age  : {self.fmt(self.w_pose_age, 1e3, " ms")}   (LiDAR pose when received)'
            f'\n  |PF - EKF|: {self.fmt(self.w_pos_err, 1e2, " cm")}   yaw {self.fmt(self.w_yaw_err, 180 / math.pi, " deg")}'
            f'\n  jumps     : {self.fmt(self.w_jump, 1e2, " cm")}')
        self.n_ekf = self.n_pose = 0
        for w in (self.w_tf_age, self.w_out_age, self.w_pos_err, self.w_yaw_err, self.w_jump, self.w_pose_age):
            w.clear()
        if self.csv:
            self.csv_file.flush()

    def summary(self):
        n = self.total['n_err']
        rms = math.sqrt(self.total['pos_err_sq'] / n) if n else float('nan')
        # print, not the logger: this runs after the context was shut down.
        print(f"[localization_monitor] summary: {self.total['ekf']} EKF outputs, {self.total['pose']} LiDAR poses, "
              f"|PF-EKF| rms {rms * 100:.1f} cm, largest jump {self.total['max_jump'] * 100:.1f} cm", flush=True)


def main():
    rclpy.init()
    node = LocalizationMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.summary()
        if node.csv:
            node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
