#!/usr/bin/env python3
"""Localisation error against the TRUTH of a synthetic bag, for the particle filter and slam_toolbox alike.

  loc_eval.py BAG RUN [RUN ...] --frame x,y,yaw [--skip 3]

RUN: a pf_replay.py .npz (the PF pose at its stamp) or a slam_loc_run.sh / slam_run.sh output directory
(log.npy: map -> base_link at every truth stamp). --frame: the map frame in the truth frame (slam_frame.py).
win5 = error left after removing each 5 s window's mean offset: what the controller sees when the path is
drawn on the same (imperfect) map. jit = error change per 0.1 s.
"""
import argparse
import math
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('runs', nargs='+')
ap.add_argument('--skip', type=float, default=3.0, help='s after the start not evaluated (convergence)')
ap.add_argument('--frame', default='0,0,0', help='x,y,yaw of the runs\' map frame in the truth frame')
a = ap.parse_args()

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
r.set_filter(rosbag2_py.StorageFilter(topics=['/truth/odom']))
tt, tx, ty, tp = [], [], [], []
while r.has_next():
    _, data, _ = r.read_next()
    m = deserialize_message(data, Odometry)
    tt.append(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec)
    tx.append(m.pose.pose.position.x)
    ty.append(m.pose.pose.position.y)
    q = m.pose.pose.orientation
    tp.append(2 * math.atan2(q.z, q.w))
tt, tx, ty, tp = map(np.array, (tt, tx, ty, tp))
tp = np.unwrap(tp)
fx, fy, fp = map(float, a.frame.split(','))


def load(f):
    if os.path.isdir(f):
        t, x, y, yaw = np.load(os.path.join(f, 'log.npy'))[:, :4].T
        cpu = math.nan
        if os.path.exists(os.path.join(f, 'cpu.txt')):
            cs, ws = map(float, open(os.path.join(f, 'cpu.txt')).read().split())
            cpu = 100.0 * cs / ws
        return t, x, y, yaw, cpu
    z = np.load(f)
    R, cols = z['rec'], list(z['cols'])
    c = {k: R[:, cols.index(k)] for k in cols}
    ts = np.where(np.isfinite(c['pub_stamp']), c['pub_stamp'], c['t'])
    # PF CPU: ms per update at ~9.7 Hz -> % of one core
    return ts, c['x'], c['y'], c['yaw'], np.nanmedian(c['cpu']) * 9.68 * 100.0


print(f'{"run":<28} {"n":>5} {"rms":>6} {"p95":>6} {"max":>6} {"along":>6} {"cross":>6} {"yaw":>5} {"win5":>5} '
      f'{"jit":>5} {"cpu%":>5}   (cm, deg)')
for f in a.runs:
    ts, x, y, yaw, cpu = load(f)
    k = (ts > tt[0] + a.skip) & (ts < tt[-1])
    ts, x, y, yaw = ts[k], x[k], y[k], yaw[k]
    x, y, yaw = fx + x * math.cos(fp) - y * math.sin(fp), fy + x * math.sin(fp) + y * math.cos(fp), yaw + fp
    gx, gy, gp = np.interp(ts, tt, tx), np.interp(ts, tt, ty), np.interp(ts, tt, tp)
    ex, ey = x - gx, y - gy
    along = ex * np.cos(gp) + ey * np.sin(gp)
    cross = -ex * np.sin(gp) + ey * np.cos(gp)
    e = np.hypot(ex, ey)
    eyaw = np.degrees(np.angle(np.exp(1j * (yaw - gp))))
    w = np.floor((ts - ts[0]) / 5.0)
    rx, ry = ex.copy(), ey.copy()
    for i in np.unique(w):
        m = w == i
        rx[m] -= ex[m].mean()
        ry[m] -= ey[m].mean()
    g = np.arange(ts[0], ts[-1], 0.1)
    jit = np.hypot(np.diff(np.interp(g, ts, ex)), np.diff(np.interp(g, ts, ey)))
    name = f.rstrip('/').split('/')[-1][:28]
    print(f'{name:<28} {len(e):>5} {100 * np.sqrt(np.mean(e ** 2)):>6.1f} {100 * np.percentile(e, 95):>6.1f} '
          f'{100 * e.max():>6.1f} {100 * np.sqrt(np.mean(along ** 2)):>6.1f} {100 * np.sqrt(np.mean(cross ** 2)):>6.1f} '
          f'{np.sqrt(np.mean(eyaw ** 2)):>5.2f} {100 * np.sqrt(np.mean(rx ** 2 + ry ** 2)):>5.1f} '
          f'{100 * np.sqrt(np.mean(jit ** 2)):>5.2f} {cpu:>5.0f}')
