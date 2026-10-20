"""Extract the topics used by the PF / EKF analysis from a rosbag into an npz.

  python3 bag_extract.py BAG_DIR OUT.npz

Needs ROS 2 + this workspace sourced (vesc_msgs). Columns: [receive time, header stamp, ...].
"""
import sys, math, numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, out = sys.argv[1], sys.argv[2]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
       rosbag2_py.ConverterOptions('cdr', 'cdr'))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = ['/pf/pose/odom', '/localization/odom', '/localization/ekf/debug', '/sensors/core',
        '/zed2/zed_node/imu/data', '/localization/twist_with_cov', '/commands/motor/speed',
        '/commands/servo/position', '/zed2/zed_node/odom', '/scan']
cls = {t: get_message(types[t]) for t in want if t in types}
d = {t: [] for t in cls}
yaw = lambda q: math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
st = lambda h: h.stamp.sec + h.stamp.nanosec*1e-9
while r.has_next():
    t, data, tn = r.read_next()
    if t not in cls: continue
    m = deserialize_message(data, cls[t]); tr = tn*1e-9
    if t in ('/pf/pose/odom', '/localization/odom', '/zed2/zed_node/odom'):
        p = m.pose.pose
        d[t].append([tr, st(m.header), p.position.x, p.position.y, yaw(p.orientation),
                     m.twist.twist.linear.x, m.twist.twist.angular.z])
    elif t == '/localization/ekf/debug':
        d[t].append([tr] + list(m.data))
    elif t == '/sensors/core':
        s = m.state
        d[t].append([tr, st(m.header), s.speed, s.current_motor, s.voltage_input, s.duty_cycle])
    elif t == '/zed2/zed_node/imu/data':
        d[t].append([tr, st(m.header), m.angular_velocity.z, m.linear_acceleration.x, m.linear_acceleration.y])
    elif t == '/localization/twist_with_cov':
        tw = m.twist.twist
        d[t].append([tr, st(m.header), tw.linear.x, tw.angular.z])
    elif t in ('/commands/motor/speed', '/commands/servo/position'):
        d[t].append([tr, m.data])
    elif t == '/scan':
        d[t].append([tr, st(m.header), len(m.ranges)])
np.savez(out, **{k.strip('/').replace('/', '__'): np.array(v) for k, v in d.items()})
for k, v in d.items():
    a = np.array(v)
    if len(a) > 1:
        print(f'{k:35s} n={len(a):6d}  {(len(a)-1)/(a[-1,0]-a[0,0]):6.1f} Hz')
    else:
        print(f'{k:35s} n={len(a)}')
