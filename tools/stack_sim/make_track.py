#!/usr/bin/env python3
"""A closed test track the car CAN drive: curvature <= 1/R_min (default 1.5 m, the car ~1.2 1/m), with
curvature ramps (no steps) and an S-bend on each straight. The original smooth_loop_path3_edit.csv has
|kappa| 2.1 at s 22-27, beyond the car: errors there are the path's, and a yaw-rate envelope fights it.

  make_track.py OUT.csv [--radius 1.5] [--ramp 0.6] [--ds 0.02]

Prints the length and where the S-bends are (for wet_s0 / wet_s1).
"""
import argparse
import math

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--radius', type=float, default=1.5)
ap.add_argument('--ramp', type=float, default=0.6, help='m over which the curvature changes')
ap.add_argument('--ds', type=float, default=0.02)
a = ap.parse_args()
k = 1.0 / a.radius


def arc(angle, sign):
    """Constant-curvature turn of `angle` rad with curvature ramps in and out (same total angle)."""
    # ramps turn ramp * k / 2 each
    body = (angle - a.ramp * k) / k
    assert body > 0, 'angle too small for the ramps'
    return [('ramp', a.ramp, 0.0, sign * k), ('const', body, sign * k, sign * k), ('ramp', a.ramp, sign * k, 0.0)]


def straight(length):
    return [('const', length, 0.0, 0.0)]


s_angle = math.radians(60)
# stadium (two 180 deg left turns) with an S-bend (right then left) on both straights: the S-bends cancel
segs = (straight(2.0) + arc(s_angle, -1) + arc(s_angle, +1) + straight(2.0) + arc(math.pi, +1) +
        straight(2.0) + arc(s_angle, -1) + arc(s_angle, +1) + straight(2.0) + arc(math.pi, +1))

xs, ys, ks, marks = [0.0], [0.0], [0.0], []
th, s = 0.0, 0.0
for i, (kind, length, k0, k1) in enumerate(segs):
    n = max(1, int(round(length / a.ds)))
    h = length / n
    if i in (1, 13):
        marks.append(s)
    for j in range(n):
        kk = k0 + (k1 - k0) * (j + 0.5) / n
        th += kk * h
        xs.append(xs[-1] + h * math.cos(th - 0.5 * kk * h))
        ys.append(ys[-1] + h * math.sin(th - 0.5 * kk * h))
        ks.append(kk)
        s += h
gap = math.hypot(xs[-1], ys[-1])
assert gap < 0.05, f'track does not close: {gap:.3f} m'
with open(a.out, 'w') as f:
    f.write('x,y,desired_speed\n')
    for x, y in zip(xs[:-1], ys[:-1]):
        f.write(f'{x:.4f},{y:.4f},0\n')
print(f'{a.out}: length {s:.2f} m, |kappa| max {max(map(abs, ks)):.3f} 1/m, closing gap {gap * 100:.1f} cm, '
      f'S-bends start at s = {marks[0]:.1f} and {marks[1]:.1f} m (each ~{2 * (s_angle / k):.1f} m long)')
