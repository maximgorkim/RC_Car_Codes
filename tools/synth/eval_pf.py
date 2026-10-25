#!/usr/bin/env python3
"""Particle-filter error against the TRUTH of a synthetic bag (make_bag.py).

  eval_pf.py BAG RUN.npz [RUN2.npz ...] [--skip 3]

RUN.npz from particle_filter/tools/pf_replay.py on that bag. The pose is compared with the truth at
the time it claims to describe: its stamp (scan_sync: the scan time; otherwise the update time,
i.e. "now"). along / cross = components along / across the true heading.
"""
import argparse
import math

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('runs', nargs='+')
ap.add_argument('--skip', type=float, default=3.0, help='s after the start not evaluated (convergence)')
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

print(f'{"run":<34} {"n":>4} {"pos_rms":>8} {"p95":>6} {"max":>6} {"along":>6} {"cross":>6} {"yaw_deg":>7} '
      f'{"jit_cm":>6} {"cpu_ms":>6}')
for f in a.runs:
    z = np.load(f)
    R, cols = z['rec'], list(z['cols'])
    c = {k: R[:, cols.index(k)] for k in cols}
    ts = np.where(np.isfinite(c['pub_stamp']), c['pub_stamp'], c['t'])
    k = (ts > tt[0] + a.skip) & (ts < tt[-1])
    ts, x, y, yaw = ts[k], c['x'][k], c['y'][k], c['yaw'][k]
    gx, gy, gp = np.interp(ts, tt, tx), np.interp(ts, tt, ty), np.interp(ts, tt, tp)
    ex, ey = x - gx, y - gy
    along = ex * np.cos(gp) + ey * np.sin(gp)
    cross = -ex * np.sin(gp) + ey * np.cos(gp)
    e = np.hypot(ex, ey)
    eyaw = np.degrees(np.angle(np.exp(1j * (yaw - gp))))
    # jitter: error change between consecutive poses (what the controller sees as noise)
    jit = np.hypot(np.diff(ex), np.diff(ey))
    print(f'{f.split("/")[-1][:34]:<34} {len(e):>4} {100 * np.sqrt(np.mean(e ** 2)):>8.2f} '
          f'{100 * np.percentile(e, 95):>6.2f} {100 * e.max():>6.1f} {100 * np.sqrt(np.mean(along ** 2)):>6.2f} '
          f'{100 * np.sqrt(np.mean(cross ** 2)):>6.2f} {np.sqrt(np.mean(eyaw ** 2)):>7.2f} '
          f'{100 * np.sqrt(np.mean(jit ** 2)):>6.2f} {1e3 * np.nanmedian(c["cpu"][k]):>6.1f}')
