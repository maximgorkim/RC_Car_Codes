"""Bag -> events.csv for ekf_replay (imu / vesc / pose / vo rows in receive order).

  python3 bag2events.py BAG OUT.csv
"""
import sys, math
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
bag, out = sys.argv[1], sys.argv[2]
r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = {'/zed2/zed_node/imu/data', '/sensors/core', '/pf/pose/odom', '/zed2/zed_node/odom'}
f = open(out, 'w'); n = 0
st = lambda h: h.stamp.sec + 1e-9 * h.stamp.nanosec
while r.has_next():
    t, b, ts = r.read_next()
    if t not in want: continue
    m = deserialize_message(b, get_message(types[t])); rc = ts * 1e-9
    if t == '/sensors/core': f.write(f'vesc,{rc:.6f},{st(m.header):.6f},{m.state.speed}\n')
    elif t.endswith('imu/data'): f.write(f'imu,{rc:.6f},{st(m.header):.6f},{m.angular_velocity.z},{m.linear_acceleration.y}\n')
    elif t == '/zed2/zed_node/odom':  # visual odometry poses (ekf_replay vo=1)
        q = m.pose.pose.orientation; yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        f.write(f'vo,{rc:.6f},{st(m.header):.6f},{m.pose.pose.position.x},{m.pose.pose.position.y},{yaw}\n')
    else:
        q = m.pose.pose.orientation; yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)); c = m.pose.covariance
        f.write(f'pose,{rc:.6f},{st(m.header):.6f},{m.pose.pose.position.x},{m.pose.pose.position.y},{yaw},{c[0]},{c[7]},{c[35]}\n')
    n += 1
print(bag, n)
