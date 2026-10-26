#!/usr/bin/env python3
# slam_frame.py MAPDIR BAG: the SLAM map's frame in the truth frame (best rigid fit of the mapping run,
# MAPDIR/log.npy) as "x,y,yaw" for eval_pf.py --frame / slam_loc_eval.py, and the true start pose of BAG
# (/pf/pose/odom) in the map frame as "x,y,yaw" for pf_replay.py --init= / slam_loc_run.sh.
import math, sys, numpy as np
import rclpy.serialization
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from nav_msgs.msg import Odometry


def map_in_truth(log):
    t, x, y, p, xt, yt, pt = np.load(log).T
    A = np.c_[x, y]; B = np.c_[xt, yt]; ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ma).T @ (B - mb)); D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T; tr = mb - R @ ma
    return tr[0], tr[1], math.atan2(R[1, 0], R[0, 0])


def start_pose(bag):
    r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('', ''))
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == '/pf/pose/odom':
            m = rclpy.serialization.deserialize_message(data, Odometry); q = m.pose.pose.orientation
            return m.pose.pose.position.x, m.pose.pose.position.y, math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)


if __name__ == '__main__':
    fx, fy, fa = map_in_truth(sys.argv[1] + '/log.npy')
    sx, sy, sa = start_pose(sys.argv[2])
    c, s = math.cos(fa), math.sin(fa)
    dx, dy = sx - fx, sy - fy
    print(f'{fx:.4f},{fy:.4f},{fa:.5f} {c * dx + s * dy:.4f},{-s * dx + c * dy:.4f},{math.atan2(math.sin(sa - fa), math.cos(sa - fa)):.5f}')
