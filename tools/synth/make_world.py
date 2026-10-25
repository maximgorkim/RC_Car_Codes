#!/usr/bin/env python3
"""Synthetic map around a path: a corridor of half-width W along the path (walls everywhere else),
optionally with boxes along the walls (features for the along-track direction). Writes the same
PGM + YAML format as the car's map (map_server, trinary, 0.05 m).

  make_world.py PATH.csv OUT_DIR/name [--width 0.8] [--boxes 2.5] [--res 0.05]

--boxes D: a 0.3 m box every D m of path, alternating sides (0 = bare corridor: along a straight
corridor the LiDAR then says nothing about the position along it, the PF's hard case).
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('out')
ap.add_argument('--width', type=float, default=0.8)
ap.add_argument('--boxes', type=float, default=2.5)
ap.add_argument('--box_size', type=float, default=0.3)
ap.add_argument('--res', type=float, default=0.05)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

d = np.genfromtxt(a.path, delimiter=',', names=True)
x, y = d[d.dtype.names[0]], d[d.dtype.names[1]]
# dense resampling of the (closed) path
seg = np.hypot(np.diff(x), np.diff(y))
s = np.r_[0, np.cumsum(seg)]
sd = np.arange(0, s[-1], a.res / 3)
xd, yd = np.interp(sd, s, x), np.interp(sd, s, y)

margin = a.width + 2.0
x0, y0 = xd.min() - margin, yd.min() - margin
W = int(np.ceil((xd.max() + margin - x0) / a.res))
H = int(np.ceil((yd.max() + margin - y0) / a.res))
# grid row 0 = y0 (bottom); flipped when written (PGM row 0 = top)
on_path = np.zeros((H, W), bool)
on_path[((yd - y0) / a.res).astype(int), ((xd - x0) / a.res).astype(int)] = True
dist = ndimage.distance_transform_edt(~on_path) * a.res
free = dist < a.width

if a.boxes > 0:
    rng = np.random.default_rng(a.seed)
    tx, ty = np.gradient(xd), np.gradient(yd)
    tn = np.hypot(tx, ty)
    tx, ty = tx / tn, ty / tn
    hb = a.box_size / 2
    side = 1.0
    for sb in np.arange(1.0, s[-1] - 1.0, a.boxes):
        i = int(sb / (a.res / 3))
        off = a.width - hb - 0.02 + rng.uniform(-0.05, 0.0)
        cx, cy = xd[i] - side * ty[i] * off, yd[i] + side * tx[i] * off
        side = -side
        # axis-aligned with the path direction
        yy, xx = np.mgrid[0:H, 0:W]
        px, py = x0 + (xx + 0.5) * a.res - cx, y0 + (yy + 0.5) * a.res - cy
        u, v = px * tx[i] + py * ty[i], -px * ty[i] + py * tx[i]
        box = (np.abs(u) < hb) & (np.abs(v) < hb)
        # never block the path itself
        box &= dist > 0.45
        free &= ~box

img = np.where(free, 254, 0).astype(np.uint8)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
Image.fromarray(np.flipud(img)).save(a.out + '.pgm')
with open(a.out + '.yaml', 'w') as f:
    f.write(f'image: {os.path.basename(a.out)}.pgm\nmode: trinary\nresolution: {a.res}\n'
            f'origin: [{x0:.4f}, {y0:.4f}, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
print(f'{a.out}.pgm {W}x{H}, corridor half-width {a.width} m, boxes every {a.boxes} m')
