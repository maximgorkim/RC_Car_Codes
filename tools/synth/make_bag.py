#!/usr/bin/env python3
"""Synthetic sensor bag from a true trajectory (stack_sim --series) and a map (make_world.py):
the same topics as the car, so the real particle filter (particle_filter/tools/pf_replay.py), the
EKF and slam_toolbox run on it, and their output can be compared with the TRUTH (/truth/odom).

  make_bag.py SERIES.csv MAP.yaml OUT_BAG [--vo_scale 1.05] [--range_std 0.015] ...

Sensors (as measured on the car, 2026-10):
  /scan        LaserScan, YDLidar G4: 900 beams over 360 deg, 9.68 Hz, frame base_link, stamp = scan
               start, every beam at its own time (motion during the scan), range noise, dropouts,
               the sectors the car body blocks and returns beyond max_range reported as 0 (as G4)
  /zed2/zed_node/odom    ZED visual odometry, 30 Hz: scale error (bags: 1.05-1.08 vs VESC), slow yaw
               drift, empty twist (as on the car)
  /zed2/zed_node/imu/data  197 Hz: yaw rate + noise (0.049 rad/s), lateral acceleration
  /sensors/core VESC 50 Hz (ERPM, 2171 per m/s / 1.04 pose units)
  /localization/twist_with_cov  twist_estimator output (VESC speed + IMU yaw rate), 197 Hz
  /pf/pose/odom  ONE message: the true start pose (pf_replay initialises from it)
  /truth/odom    ground truth of base_link, 50 Hz (evaluation only)
"""
import argparse
import math
import os
import shutil

import numpy as np
import rclpy.serialization as ser
import rosbag2_py
import yaml
from PIL import Image

import range_libc
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Imu, LaserScan
from vesc_msgs.msg import VescStateStamped

ap = argparse.ArgumentParser()
ap.add_argument('series')
ap.add_argument('map')
ap.add_argument('out')
ap.add_argument('--beams', type=int, default=900)
ap.add_argument('--scan_hz', type=float, default=9.68)
ap.add_argument('--max_range', type=float, default=12.0)
ap.add_argument('--range_std', type=float, default=0.015, help='m, plus range_rel * range')
ap.add_argument('--range_rel', type=float, default=0.005)
ap.add_argument('--dropout', type=float, default=0.04)
ap.add_argument('--blocked', default='120:170:0.65,-180:-165:0.4',
                help='deg_from:deg_to:probability sectors returning 0 (car body; from the 2026-10-19 bag)')
ap.add_argument('--vo_scale', type=float, default=1.05)
ap.add_argument('--vo_yaw_rw', type=float, default=0.003, help='rad/sqrt(s) VO yaw drift')
ap.add_argument('--gyro_std', type=float, default=0.049)
ap.add_argument('--t0', type=float, default=1.0, help='s of the series to skip (car at rest)')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--no_skew', action='store_true', help='all beams at the scan middle (an ideal, deskewed scan)')
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

# ---------------------------------------------------------------- truth, interpolated
d = np.genfromtxt(a.series, delimiter=',', names=True)
d = d[d['t'] >= a.t0]
T, BX, BY, PSI, V, R = d['t'], d['bx'], d['by'], np.unwrap(d['psi']), d['v'], d['r']
def truth(t):
    return np.interp(t, T, BX), np.interp(t, T, BY), np.interp(t, T, PSI), np.interp(t, T, V), np.interp(t, T, R)

# ---------------------------------------------------------------- map for ray casting
m = yaml.safe_load(open(a.map))
img = np.array(Image.open(os.path.join(os.path.dirname(a.map), m['image'])).convert('L'), dtype=float)
p = (255.0 - img) / 255.0
grid = np.full(img.shape, -1, dtype=np.int8)
grid[p > m['occupied_thresh']] = 100
grid[p < m['free_thresh']] = 0
g = OccupancyGrid()
g.info.resolution = float(m['resolution'])
g.info.width, g.info.height = img.shape[1], img.shape[0]
g.info.origin.position.x, g.info.origin.position.y = float(m['origin'][0]), float(m['origin'][1])
g.info.origin.orientation.w = 1.0
g.data = np.flipud(grid).flatten().tolist()
caster = range_libc.PyRayMarching(range_libc.PyOMap(g), int(a.max_range / g.info.resolution))

# ---------------------------------------------------------------- bag
if os.path.exists(a.out):
    shutil.rmtree(a.out)
w = rosbag2_py.SequentialWriter()
w.open(rosbag2_py.StorageOptions(uri=a.out, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
for name, typ in [('/scan', 'sensor_msgs/msg/LaserScan'), ('/zed2/zed_node/odom', 'nav_msgs/msg/Odometry'),
                  ('/zed2/zed_node/imu/data', 'sensor_msgs/msg/Imu'), ('/sensors/core', 'vesc_msgs/msg/VescStateStamped'),
                  ('/localization/twist_with_cov', 'geometry_msgs/msg/TwistWithCovarianceStamped'),
                  ('/pf/pose/odom', 'nav_msgs/msg/Odometry'), ('/truth/odom', 'nav_msgs/msg/Odometry')]:
    w.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format='cdr'))

T_BASE = 1.8e9  # bag clock (as a wall time)
def stamp(msg, t):
    tt = T_BASE + t
    msg.header.stamp.sec = int(tt)
    msg.header.stamp.nanosec = int(round((tt - int(tt)) * 1e9)) % 1000000000
events = []  # (receive time, topic, message)
def quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)
def odom(t, x, y, yaw, frame='odom'):
    o = Odometry()
    stamp(o, t)
    o.header.frame_id, o.child_frame_id = frame, 'base_link'
    o.pose.pose.position.x, o.pose.pose.position.y = float(x), float(y)
    q = quat(yaw)
    o.pose.pose.orientation.z, o.pose.pose.orientation.w = q[2], q[3]
    return o

t_start, t_end = T[0] + 0.2, T[-1] - 0.3
x0, y0, p0, _, _ = truth(t_start)
init = odom(t_start, x0, y0, p0)
init.pose.covariance[0] = init.pose.covariance[7] = 1e-4
events.append((t_start, '/pf/pose/odom', init))

# truth 50 Hz
for t in np.arange(t_start, t_end, 0.02):
    x, y, ps, v, r = truth(t)
    o = odom(t, x, y, ps)
    o.twist.twist.linear.x, o.twist.twist.angular.z = float(v), float(r)
    events.append((t, '/truth/odom', o))

# LiDAR
N = a.beams
ang = np.linspace(-math.pi, math.pi, N, endpoint=False).astype(np.float32)
period = 1.0 / a.scan_hz
blocked = np.zeros(N)
for sec in a.blocked.split(','):
    if sec:
        lo, hi, pr = map(float, sec.split(':'))
        blocked[(np.degrees(ang) >= lo) & (np.degrees(ang) < hi)] = pr
t = t_start + 0.05
q = np.zeros((N, 3), np.float32)
out = np.zeros(N, np.float32)
n_scan = 0
while t + period < t_end:
    tb = t + np.arange(N) * (period / N)  # time of every beam
    if a.no_skew:
        tb = np.full(N, t + 0.5 * period)
    x, y, ps, _, _ = truth(tb)
    q[:, 0], q[:, 1], q[:, 2] = x, y, ps + ang
    caster.calc_range_many(q, out)
    rr = out.astype(float)
    rr = rr + rng.normal(0.0, a.range_std + a.range_rel * rr)
    bad = (out >= a.max_range - 1e-3) | (rng.random(N) < a.dropout) | (rng.random(N) < blocked)
    rr[bad] = 0.0
    msg = LaserScan()
    stamp(msg, t)
    msg.header.frame_id = 'base_link'
    msg.angle_min, msg.angle_max = float(ang[0]), float(ang[-1])
    msg.angle_increment = float(2 * math.pi / N)
    msg.time_increment, msg.scan_time = period / N, period
    msg.range_min, msg.range_max = 0.01, 64.0
    msg.ranges = rr.astype(np.float32).tolist()
    events.append((t + period + 0.005, '/scan', msg))  # published when complete
    t += period
    n_scan += 1

# ZED visual odometry 30 Hz: integrate the true motion with scale error and yaw drift
vx, vy, vyaw, yaw_err = 0.0, 0.0, 0.0, 0.0
tp = t_start
px, py, pp, _, _ = truth(tp)
for t in np.arange(t_start, t_end, 1 / 30.0):
    x, y, ps, _, _ = truth(t)
    dx, dy = x - px, y - py
    c, s = math.cos(pp), math.sin(pp)  # true motion in the car frame of the previous sample
    fwd, lat = c * dx + s * dy, -s * dx + c * dy
    dyaw = ps - pp
    yaw_err += a.vo_yaw_rw * math.sqrt(1 / 30.0) * rng.normal()
    cv, sv = math.cos(vyaw), math.sin(vyaw)
    vx += a.vo_scale * (cv * fwd - sv * lat)
    vy += a.vo_scale * (sv * fwd + cv * lat)
    vyaw += dyaw + a.vo_yaw_rw * math.sqrt(1 / 30.0) * rng.normal()
    px, py, pp = x, y, ps
    events.append((t + 0.03, '/zed2/zed_node/odom', odom(t, vx, vy, vyaw)))

# IMU + twist_estimator 197 Hz, VESC 50 Hz
erpm_per_mps, pose_per_vesc = 2171.0, 1.04
for t in np.arange(t_start, t_end, 1 / 197.0):
    _, _, _, v, r = truth(t)
    im = Imu()
    stamp(im, t)
    im.header.frame_id = 'zed2_imu_link'
    wz = r + a.gyro_std * rng.normal()
    im.angular_velocity.z = float(wz)
    im.linear_acceleration.y = float(v * r + rng.normal())
    im.linear_acceleration.z = 9.81
    events.append((t + 0.005, '/zed2/zed_node/imu/data', im))
    tw = TwistWithCovarianceStamped()
    stamp(tw, t)
    tw.header.frame_id = 'base_link'
    tw.twist.twist.linear.x = float(v / pose_per_vesc * erpm_per_mps / 2080.0)  # twist_estimator's 2080 ERPM per m/s
    tw.twist.twist.angular.z = float(wz)
    tw.twist.covariance[0], tw.twist.covariance[35] = 0.01, 0.0004
    events.append((t + 0.006, '/localization/twist_with_cov', tw))
for t in np.arange(t_start, t_end, 0.02):
    _, _, _, v, _ = truth(t)
    vs = VescStateStamped()
    stamp(vs, t)
    vs.state.speed = float(v / pose_per_vesc * erpm_per_mps)
    events.append((t + 0.002, '/sensors/core', vs))

events.sort(key=lambda e: e[0])
for tr, topic, msg in events:
    w.write(topic, ser.serialize_message(msg), int((T_BASE + tr) * 1e9))
del w
print(f'{a.out}: {t_end - t_start:.1f} s, {n_scan} scans, {len(events)} messages')
