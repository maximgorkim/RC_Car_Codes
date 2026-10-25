#!/usr/bin/env python3
# slam_log.py OUT.npy: logs map->base_link (TF) at every /truth/odom stamp (sim time) until Ctrl+C
import sys, math, rclpy, numpy as np
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
class L(Node):
    def __init__(s):
        super().__init__('slamlog', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        s.buf = Buffer(); s.tl = TransformListener(s.buf, s)
        s.rows = []; s.pending = []
        s.create_subscription(Odometry, '/truth/odom', s.cb, 100)
    def cb(s, m):
        s.pending.append(m)
        keep = []
        for o in s.pending:
            t = Time.from_msg(o.header.stamp)
            if s.buf.can_transform('map', 'base_link', t):
                tr = s.buf.lookup_transform('map', 'base_link', t).transform
                q = tr.rotation; yaw = math.atan2(2*(q.w*q.z), 1-2*q.z*q.z)
                qo = o.pose.pose.orientation; yt = math.atan2(2*(qo.w*qo.z), 1-2*qo.z*qo.z)
                s.rows.append((t.nanoseconds*1e-9, tr.translation.x, tr.translation.y, yaw,
                               o.pose.pose.position.x, o.pose.pose.position.y, yt))
            elif (Time.from_msg(m.header.stamp) - t).nanoseconds < 1e9:
                keep.append(o)
        s.pending = keep
rclpy.init(); n = L()
try:
    rclpy.spin(n)
except (KeyboardInterrupt, Exception):
    pass
np.save(sys.argv[1], np.array(n.rows))
