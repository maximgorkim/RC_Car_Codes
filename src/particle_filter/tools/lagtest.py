"""Pose lag of the PF vs the integrated ZED gyro yaw during a turn window.

  python3 lagtest.py BAG.npz T_FROM T_TO REPLAY.npz [REPLAY2.npz ...]

Prints, for the car's PF (as recorded) and each replay, the delay that best aligns the
PF heading with the gyro: at publication time and relative to the published stamp.
"""
import sys, numpy as np
z=np.load(sys.argv[1]); zo=z['zed2__zed_node__odom']; pf=z['pf__pose__odom']; t0=pf[0,0]; lo,hi=float(sys.argv[2]),float(sys.argv[3])
imu=z["zed2__zed_node__imu__data"]; zt=imu[:,1]; zyaw=np.concatenate([[0],np.cumsum(imu[1:,2]*np.diff(zt))])
def lag(ts, yaw):
    m=(ts-t0>lo)&(ts-t0<hi); ts,yaw=ts[m],np.unwrap(yaw[m]); best=None
    for tau in np.arange(-0.1,0.4,0.005):
        ref=np.interp(ts-tau,zt,zyaw); e=yaw-ref; e=e-e.mean(); rms=np.sqrt(np.mean(e**2))
        if best is None or rms<best[1]: best=(tau,rms)
    return best
# car PF as received: header = now()
tau,rms=lag(pf[:,1],pf[:,4]); print(f'{"arac PF (header=now)":34s} gecikme {tau*1e3:5.0f} ms  (yaw rms {np.degrees(rms):.2f} deg)')
for p in sys.argv[4:]:
    R=np.load(p)['rec']
    for col,lab in [(0,'yayin ani'),(13,'yayinlanan damga')]:
        ts=R[:,col]
        if np.all(np.isnan(ts)): continue
        tau,rms=lag(ts,R[:,5]); print(f'{p.split("/")[-1]+" "+lab:34s} gecikme {tau*1e3:5.0f} ms  (yaw rms {np.degrees(rms):.2f} deg)')
