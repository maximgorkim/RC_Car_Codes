"""Which way must a scan be deskewed? Turning scans of a real bag are fitted to the map (median
distance of beam ends to the nearest wall, best of a small pose search around the car PF) raw,
with the rotation deskewed assuming beams in angle order in time (fwd) and in reverse (rev).
fwd clearly below raw: scan_deskew is right for this LiDAR. Needs a map that fits (< ~10 cm).

  WMIN=0.5 python3 deskew_check.py BAG MAP.yaml T_END
"""
import sys, numpy as np, yaml, os
from PIL import Image
from scipy import ndimage
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
bag, mapy, t_end = sys.argv[1], sys.argv[2], float(sys.argv[3])
m = yaml.safe_load(open(mapy)); img = np.array(Image.open(os.path.join(os.path.dirname(mapy), m['image'])).convert('L'), float)
occ = np.flipud(img < 100); res = m['resolution']; ox, oy = m['origin'][:2]
dist = ndimage.distance_transform_edt(~occ) * res
r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
scans, tw, pf = [], [], []
t0 = None
while r.has_next():
    t, b, ts = r.read_next()
    if t0 is None: t0 = ts
    if (ts - t0) * 1e-9 > t_end: break
    if t == '/scan':
        mm = deserialize_message(b, get_message(types[t])); scans.append((mm.header.stamp.sec + 1e-9 * mm.header.stamp.nanosec, np.array(mm.ranges), mm.angle_min, mm.angle_max, mm.scan_time))
    elif t == '/zed2/zed_node/imu/data':
        mm = deserialize_message(b, get_message(types[t])); tw.append((mm.header.stamp.sec + 1e-9 * mm.header.stamp.nanosec, mm.angular_velocity.z))
    elif t == '/pf/pose/odom':
        mm = deserialize_message(b, get_message(types[t])); q = mm.pose.pose.orientation
        pf.append((ts * 1e-9, mm.pose.pose.position.x, mm.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)))
tw, pf = np.array(tw), np.array(pf)
def score(x, y, th, ang, rr, dth):
    ok = rr > 0.05
    a = ang[ok] + th + dth[ok]; px = x + rr[ok] * np.cos(a); py = y + rr[ok] * np.sin(a)
    i = ((py - oy) / res).astype(int); j = ((px - ox) / res).astype(int)
    k = (i >= 0) & (i < dist.shape[0]) & (j >= 0) & (j < dist.shape[1])
    return np.median(dist[i[k], j[k]])
# only rotation is tested (yaw rate from the IMU): the rotation part of the skew dominates the angle
res_ = {'raw': [], 'fwd': [], 'rev': []}
for (ts, rr, a0, a1, st) in scans:
    w = np.interp(ts + st / 2, tw[:, 0], tw[:, 1])
    if abs(w) < float(os.environ.get("WMIN", "0.5")): continue
    # best yaw/position from a small local search around the car PF pose (pose quality aside)
    x, y, th = np.interp(ts + st / 2, pf[:, 0] - 0.2, pf[:, 1]), np.interp(ts + st / 2, pf[:, 0] - 0.2, pf[:, 2]), np.interp(ts + st / 2, pf[:, 0] - 0.2, np.unwrap(pf[:, 3]))
    n = len(rr); ang = np.linspace(a0, a1, n); tb = np.arange(n) / n * st - st / 2
    for name, dth in [('raw', np.zeros(n)), ('fwd', w * tb), ('rev', -w * tb)]:
        best = min(score(x + dx, y + dy, th + dd, ang, rr, dth) for dx in (-0.1, 0, 0.1) for dy in (-0.1, 0, 0.1) for dd in np.linspace(-0.08, 0.08, 9))
        res_[name].append(best)
for k, v in res_.items(): print(k, 'turning scans', len(v), 'median wall distance of beam ends (cm): %.2f' % (100 * np.median(v)))
