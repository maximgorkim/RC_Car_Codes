#!/usr/bin/env python3
"""Look at a stack_sim run: the path, the car's true trajectory coloured by its lateral error,
and the signals over time. With --play the car drives the run in a window (x speed).

  build/stack_sim/stack_sim PATH.csv ... --series run.csv
  python3 tools/stack_sim/view.py PATH.csv run.csv [--play] [--speed 4] [--png out.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('series')
ap.add_argument('--play', action='store_true')
ap.add_argument('--speed', type=float, default=4.0, help='playback speed (x real time)')
ap.add_argument('--png', help='save the figure instead of opening a window')
a = ap.parse_args()

p = np.genfromtxt(a.path, delimiter=',', names=True)
px, py = p[p.dtype.names[0]], p[p.dtype.names[1]]
d = np.genfromtxt(a.series, delimiter=',', names=True)
t, x, y, psi = d['t'], d['bx'], d['by'], d['psi']
el = d['e_lat']

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(1, 2, 1)
ax.plot(px, py, color='0.7', lw=6, solid_capstyle='round', label='path')
sc = ax.scatter(x, y, c=np.abs(el) * 100, s=4, cmap='plasma', vmin=0, vmax=max(10, np.percentile(np.abs(el) * 100, 99)))
fig.colorbar(sc, ax=ax, label='|lateral error| [cm]', shrink=0.7)
ax.set_aspect('equal')
ax.set_title('top view: true trajectory (base_link)')
ax.legend(loc='upper right')

signals = [('e_lat', 100, 'lateral error [cm]'), ('v', 1, 'speed [m/s]'), ('delta_cmd', 1, 'steering cmd [rad]')]
if 'grip' in d.dtype.names and np.ptp(d['grip']) > 0:
    signals.append(('grip', 1, 'grip estimate [g]'))
if 'ekf_err' in d.dtype.names:
    signals.append(('ekf_err', 100, 'EKF position error [cm]'))
axs = []
for i, (k, sc_, lab) in enumerate(signals):
    axi = fig.add_subplot(len(signals), 2, 2 * i + 2, sharex=axs[0] if axs else None)
    axi.plot(t, d[k] * sc_, lw=0.8)
    axi.set_ylabel(lab, fontsize=8)
    axi.grid(alpha=0.3)
    axs.append(axi)
axs[-1].set_xlabel('t [s]')
fig.tight_layout()

if a.play:
    car, = ax.plot([], [], 'o', color='tab:green', ms=10, zorder=5)
    head, = ax.plot([], [], '-', color='tab:green', lw=3, zorder=5)
    cursors = [axi.axvline(t[0], color='tab:green', lw=1) for axi in axs]
    step = max(1, int(round(a.speed * 0.04 / np.median(np.diff(t)))))

    def frame(i):
        i = min(i * step, len(t) - 1)
        car.set_data([x[i]], [y[i]])
        head.set_data([x[i], x[i] + 0.4 * np.cos(psi[i])], [y[i], y[i] + 0.4 * np.sin(psi[i])])
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        ax.set_title(f'top view   t = {t[i]:5.1f} s   v = {d["v"][i]:.2f} m/s   e_lat = {100 * el[i]:+.1f} cm')
        return [car, head, *cursors]

    anim = FuncAnimation(fig, frame, frames=len(t) // step + 1, interval=40, blit=False, repeat=False)

if a.png:
    fig.savefig(a.png, dpi=110)
else:
    plt.show()
