#!/usr/bin/env python3
"""Live terminal dashboard while the car drives (sim or real car).

  ros2 run rc_localization live_status.py --ros-args -p use_sim_time:=true

Shows speed, path-tracking error, MPC and EKF health in one block that is
redrawn 5 times a second. Every topic is optional; missing ones show '-'.
In Gazebo (/sim/ground_truth/odom) it also shows the EKF and PF error
against the true pose.
"""
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # Foxy
    ExternalShutdownException = KeyboardInterrupt


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def fmt(v, spec, scale=1.0):
    return '-' if v is None else format(v * scale, spec)


class LiveStatus(Node):
    def __init__(self):
        super().__init__('live_status')
        self.ekf = self.pf = self.truth = self.mpc = self.dbg = None
        self.wall = {}            # topic -> wall time of the last message
        self.ekf_count = 0
        self.ekf_rate = None
        self.rate_t0 = time.monotonic()
        self.max_ekf_err = 0.0
        self.max_e_lat = 0.0
        self.create_subscription(Odometry, '/localization/odom', self.on_ekf, 10)
        self.create_subscription(Odometry, '/pf/pose/odom', lambda m: self.keep('pf', m), 10)
        self.create_subscription(Odometry, '/sim/ground_truth/odom', lambda m: self.keep('truth', m), 10)
        self.create_subscription(Float64MultiArray, '/mpc/state', lambda m: self.keep('mpc', m), 10)
        self.create_subscription(Float64MultiArray, '/localization/ekf/debug', lambda m: self.keep('dbg', m), 10)
        self.create_timer(0.2, self.draw)

    def keep(self, name, msg):
        setattr(self, name, msg)
        self.wall[name] = time.monotonic()

    def on_ekf(self, msg):
        self.keep('ekf', msg)
        self.ekf_count += 1

    def fresh(self, name, max_age=0.5):
        return name in self.wall and time.monotonic() - self.wall[name] < max_age

    def err_to_truth(self, msg):
        if msg is None or not self.fresh('truth'):
            return None
        p, t = msg.pose.pose.position, self.truth.pose.pose.position
        return math.hypot(p.x - t.x, p.y - t.y)

    def draw(self):
        now = time.monotonic()
        if now - self.rate_t0 >= 1.0:
            self.ekf_rate = self.ekf_count / (now - self.rate_t0)
            self.ekf_count, self.rate_t0 = 0, now

        speed = None
        if self.fresh('truth'):
            tw = self.truth.twist.twist.linear
            speed = math.hypot(tw.x, tw.y)
        elif self.fresh('ekf'):
            speed = self.ekf.twist.twist.linear.x

        m = self.mpc.data if self.fresh('mpc') else None
        d = self.dbg.data if self.fresh('dbg') else None
        ekf_err = self.err_to_truth(self.ekf if self.fresh('ekf') else None)
        pf_err = self.err_to_truth(self.pf if self.fresh('pf', 1.0) else None)
        if ekf_err is not None:
            self.max_ekf_err = max(self.max_ekf_err, ekf_err)
        if m is not None:
            self.max_e_lat = max(self.max_e_lat, abs(m[1]))

        pf_ekf = None
        if self.fresh('ekf') and self.fresh('pf', 1.0):
            a, b = self.ekf.pose.pose.position, self.pf.pose.pose.position
            pf_ekf = math.hypot(a.x - b.x, a.y - b.y)

        mpc_state = 'CALISIYOR' if m is not None else 'YOK / DURDU'
        ekf_state = 'CALISIYOR' if self.fresh('ekf') else 'YOK'
        lines = [
            '=== ARAC CANLI DURUM ===  (Ctrl+C: cikis)',
            '',
            f'Hiz          : {fmt(speed, "5.2f")} m/s   hedef {fmt(m[8] if m else None, "5.2f")}'
            f'   komut {fmt(m[9] if m else None, "5.2f")} m/s',
            f'Direksiyon   : {fmt(m[10] if m else None, "+6.1f", 180 / math.pi)} deg',
            '',
            f'MPC          : {mpc_state}   cozum {fmt(m[11] if m else None, "4.1f")} ms',
            f'Yol hatasi   : yanal {fmt(m[1] if m else None, "+6.1f", 100)} cm (en kotu {self.max_e_lat * 100:5.1f})'
            f'   yon {fmt(m[2] if m else None, "+5.1f", 180 / math.pi)} deg',
            '',
            f'EKF          : {ekf_state}   {fmt(self.ekf_rate, "4.0f")} Hz'
            f'   belirsizlik {fmt(d[9] if d else None, "4.1f", 100)} cm',
            f'|PF - EKF|   : {fmt(pf_ekf, "5.1f", 100)} cm   (PF ~0.2 s geriden gelir, hizla buyur)',
            f'Ret / reset  : {fmt(d[7] if d else None, ".0f")} / {fmt(d[8] if d else None, ".0f")}'
            f'   kayma {fmt(d[11] if d else None, "+5.2f")} m/s',
        ]
        if self.fresh('truth'):
            lines += [
                '',
                f'Gercege gore : EKF {fmt(ekf_err, "5.1f", 100)} cm (en kotu {self.max_ekf_err * 100:5.1f})'
                f'   PF {fmt(pf_err, "5.1f", 100)} cm',
            ]
        print('\033[H\033[J' + '\n'.join(lines), flush=True)


def main():
    rclpy.init()
    node = LiveStatus()
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
