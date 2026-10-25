#!/usr/bin/env python3
# slam_eval.py OUT...: slam map->base_link vs truth; raw and after the best rigid 2D alignment (SLAM map frame is its own)
import sys, numpy as np
def align(x, y, p, xt, yt, pt):
    A = np.c_[x, y]; B = np.c_[xt, yt]; ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ma).T @ (B - mb)); D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T; Ar = (R @ (A - ma).T).T + mb; rot = np.arctan2(R[1, 0], R[0, 0])
    return Ar[:, 0], Ar[:, 1], p + rot, np.degrees(rot)
for d in sys.argv[1:]:
    try: r = np.load(d + '/log.npy')
    except Exception: r = []
    if len(r) == 0: print(d, 'empty'); continue
    t, x, y, p, xt, yt, pt = r.T
    e = np.hypot(x - xt, y - yt)
    xa, ya, pa, rot = align(x, y, p, xt, yt, pt)
    ea = np.hypot(xa - xt, ya - yt); eya = np.degrees(np.angle(np.exp(1j * (pa - pt))))
    q = len(ea) // 4
    print(f'{d}: n={len(e)} raw mean {e.mean()*100:.1f} cm | aligned (rot {rot:.1f} deg) mean {ea.mean()*100:.1f} p95 {np.percentile(ea,95)*100:.1f} '
          f'max {ea.max()*100:.1f} cm, yaw rms {np.sqrt((eya**2).mean()):.2f} deg | quarters ' + ' '.join(f'{ea[i*q:(i+1)*q].mean()*100:.0f}' for i in range(4)))
