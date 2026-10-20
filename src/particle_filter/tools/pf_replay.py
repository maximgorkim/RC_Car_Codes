"""Offline replay of the real particle_filter node over a rosbag.

Feeds /scan and the motion topic to the node's own callbacks in bag order, with
the node clock driven by the bag receive time. No map_server: the map is read
from the yaml/pgm. Records every MCL update (time, scan used, pose, CPU time).

  python3 pf_replay.py BAG OUT.npz [--set name:=value ...] [--init x,y,yaw] [--t0 s --t1 s]
"""
import argparse, math, os, sys, time
import numpy as np, yaml
from PIL import Image

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/particle_filter
sys.path.insert(0, PKG)   # the source particle_filter module, not the installed one

import rclpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import range_libc
from particle_filter import particle_filter as pfmod
from particle_filter import utils as Utils

ap = argparse.ArgumentParser()
ap.add_argument('bag'); ap.add_argument('out')
ap.add_argument('--config', default=os.path.join(PKG, 'config/localize.yaml'))
ap.add_argument('--map', default=os.path.join(PKG, 'maps/my_map__.yaml'))
ap.add_argument('--set', nargs='*', default=[], help='extra ROS params name:=value')
ap.add_argument('--init', default=None, help='x,y,yaw; default: first /pf/pose/odom of the bag')
ap.add_argument('--t0', type=float, default=0.0); ap.add_argument('--t1', type=float, default=1e9)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--scan_period', type=float, default=0.0,
                help='>0: keep only scans spaced at least this far apart (e.g. emulate a slower lidar)')
args = ap.parse_args()
np.random.seed(args.seed)


def load_grid(path):
    m = yaml.safe_load(open(path))
    img = np.array(Image.open(os.path.join(os.path.dirname(path), m['image'])).convert('L'), dtype=float)
    p = (255.0 - img) / 255.0 if not m.get('negate', 0) else img / 255.0
    data = np.full(img.shape, -1, dtype=np.int8)
    data[p > m['occupied_thresh']] = 100
    data[p < m['free_thresh']] = 0
    g = OccupancyGrid()
    g.header.frame_id = 'odom'
    g.info.resolution = float(m['resolution'])
    g.info.width, g.info.height = img.shape[1], img.shape[0]
    g.info.origin.position.x, g.info.origin.position.y = float(m['origin'][0]), float(m['origin'][1])
    g.info.origin.orientation.w = 1.0
    g.data = np.flipud(data).flatten().tolist()   # map_server: row 0 = bottom
    return g


GRID = load_grid(args.map)


def get_omap_from_file(self):
    self.map_info = GRID.info
    oMap = range_libc.PyOMap(GRID)
    self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)
    rm = self.WHICH_RM
    if rm == 'bl': self.range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
    elif 'cddt' in rm:
        self.range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        if rm == 'pcddt': self.range_method.prune()
    elif rm == 'rm': self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
    elif rm == 'glt': self.range_method = range_libc.PyGiantLUTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
    else: raise RuntimeError('range_method %s not available offline (no CUDA here)' % rm)
    arr = np.array(GRID.data).reshape((GRID.info.height, GRID.info.width))
    self.permissible_region = np.zeros_like(arr, dtype=bool)
    self.permissible_region[arr == 0] = 1
    self.map_initialized = True


class FakeClock:
    ns = 0
    def now(self): return Time(nanoseconds=FakeClock.ns)


pfmod.ParticleFiler.get_omap = get_omap_from_file
pfmod.ParticleFiler.get_clock = lambda self: FakeClock()

ros_args = ['--ros-args', '--params-file', args.config, '-p', 'range_method:=rm', '-p', 'viz:=0']
for s in args.set: ros_args += ['-p', s]
rclpy.init(args=ros_args)
pf = pfmod.ParticleFiler()

# record each MCL update: wrap MCL (timing) and publish_tf (pose + stamp)
REC = []
cur = {}
orig_mcl, orig_pub = pf.MCL, pf.publish_tf
def mcl(a, o):
    t = time.perf_counter(); orig_mcl(a, o); cur['cpu'] = time.perf_counter() - t
    cur['action'] = np.array(a, dtype=float)
def pub(pose, stamp=None):
    cov = np.cov(pf.particles.T, aweights=pf.weights, ddof=0)
    st = stamp.sec + 1e-9 * stamp.nanosec if stamp is not None else np.nan
    REC.append([FakeClock.ns * 1e-9, cur.get('scan_rx', np.nan), cur.get('scan_st', np.nan),
                pose[0], pose[1], pose[2], cur.get('cpu', np.nan), cov[0, 0], cov[1, 1], cov[2, 2],
                *cur.get('action', [np.nan] * 3), st, cur.get('scan_meas', np.nan)])
pf.MCL, pf.publish_tf = mcl, pub

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
motion_topic = pf.TWIST_TOPIC if pf.MOTION_SOURCE == 'twist' else pf.get_parameter('odometry_topic').value
topics = {'/scan': pf.lidarCB, motion_topic: pf.twistCB if pf.MOTION_SOURCE == 'twist' else pf.odomCB}
cls = {t: get_message(types[t]) for t in list(topics) + ['/pf/pose/odom'] if t in types}

t_first = None; inited = args.init is not None
if inited:
    x, y, th = map(float, args.init.split(','))
last_scan_rx = -1e9
n_scan = 0
while r.has_next():
    topic, data, tn = r.read_next()
    if topic not in cls: continue
    if t_first is None: t_first = tn * 1e-9
    rel = tn * 1e-9 - t_first
    if rel > args.t1: break
    msg = deserialize_message(data, cls[topic])
    if topic == '/pf/pose/odom':
        if not inited and rel >= args.t0:
            p = msg.pose.pose; x, y, th = p.position.x, p.position.y, Utils.quaternion_to_angle(p.orientation)
            inited = True
            ps = Pose(); ps.position.x, ps.position.y = x, y; ps.orientation = Utils.angle_to_quaternion(th)
            pf.initialize_particles_pose(ps)
        continue
    if rel < args.t0: continue
    if inited and args.init is not None and n_scan == 0 and not getattr(pf, '_init_done', False):
        ps = Pose(); ps.position.x, ps.position.y = x, y; ps.orientation = Utils.angle_to_quaternion(th)
        pf.initialize_particles_pose(ps); pf._init_done = True
    if not inited: continue
    FakeClock.ns = tn
    if topic == '/scan':
        if args.scan_period > 0 and tn * 1e-9 - last_scan_rx < args.scan_period - 0.01: continue
        last_scan_rx = tn * 1e-9; n_scan += 1
        cur['scan_rx'] = tn * 1e-9; cur['scan_st'] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cur['scan_meas'] = cur['scan_st'] + 0.5 * msg.scan_time
    topics[topic](msg)

R = np.array(REC)
np.savez(args.out, rec=R, t_first=t_first, cols=np.array(
    ['t', 'scan_rx', 'scan_stamp', 'x', 'y', 'yaw', 'cpu', 'cxx', 'cyy', 'cthth', 'ax', 'ay', 'ath', 'pub_stamp', 'scan_meas']))
print(f'updates {len(R)}, scans fed {n_scan}, CPU per MCL med {np.nanmedian(R[:,6])*1e3:.1f} ms '
      f'p95 {np.nanpercentile(R[:,6],95)*1e3:.1f} ms')
