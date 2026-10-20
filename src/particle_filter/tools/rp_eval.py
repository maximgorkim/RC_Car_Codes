"""Timing / consistency metrics of pf_replay outputs vs the bag (car PF, wheels, ZED odom).

  python3 rp_eval.py BAG.npz T_END REPLAY.npz [REPLAY2.npz ...]
"""
import sys, numpy as np
wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi
def metrics(rp_path, npz_path, t1=48.0, name=None):
    d=np.load(rp_path); R=d['rec']; z=np.load(npz_path)
    pf, core = z['pf__pose__odom'], z['sensors__core']; t0=pf[0,0]
    R=R[(R[:,0]-t0)<=t1]
    v=np.interp(R[:,0],core[:,0],core[:,2])/2080.
    cum=np.concatenate([[0],np.cumsum(np.abs(core[1:,2]/2080.)*np.diff(core[:,0]))]); cw=np.interp(R[:,0],core[:,0],cum)
    # scan reuse
    u,c=np.unique(R[:,2],return_counts=True); reuse=(c-1).sum()/len(R)
    # latency scan stamp -> update (no compute time)
    lat=(R[:,0]-R[:,2])*1e3
    dt=np.diff(R[:,0])*1e3
    # fidelity vs car PF
    cx=np.interp(R[:,0],pf[:,0],pf[:,2]); cy=np.interp(R[:,0],pf[:,0],pf[:,3]); dcar=np.hypot(R[:,3]-cx,R[:,4]-cy)
    # 2s path ratio (moving, straight)
    rate=(len(R)-1)/(R[-1,0]-R[0,0]); k=max(1,int(round(2*rate)))
    ch=np.hypot(R[k:,3]-R[:-k,3],R[k:,4]-R[:-k,4]); wd=cw[k:]-cw[:-k]; turn=np.abs(wrap(R[k:,5]-R[:-k,5])); m=(wd>1.0)&(turn<0.1)
    st=np.abs(v[1:])<0.015; stp=np.hypot(np.diff(R[:,3]),np.diff(R[:,4]))
    # "jerk": deviation of each step from wheel distance travelled, while moving
    mv=np.abs(v[1:])>0.3; dev=np.abs(stp-np.diff(cw))
    # consistency vs ZED odometry: relative motion between consecutive poses, both
    # taken at the scan measurement time (where the pose should be valid)
    zo=z['zed2__zed_node__odom']; tm=R[:,14] if R.shape[1]>14 else R[:,2]+0.07
    zt=zo[:,1]; zx=np.interp(tm,zt,zo[:,2]); zy=np.interp(tm,zt,zo[:,3]); zth=np.interp(tm,zt,np.unwrap(zo[:,4]))
    def rel(x,y,th):
        dx,dy=np.diff(x),np.diff(y); c,s=np.cos(th[:-1]),np.sin(th[:-1]); return c*dx+s*dy, -s*dx+c*dy, wrap(np.diff(th))
    pa,pc,pt=rel(R[:,3],R[:,4],R[:,5]); za,zc,zt_=rel(zx,zy,zth); mvm=np.abs(v[1:])>0.3
    ea=(pa-za)[mvm]*100; ec=(pc-zc)[mvm]*100
    return dict(cons_a=np.sqrt(np.mean(ea**2)), cons_a_med=np.median(np.abs(ea)), cons_c=np.sqrt(np.mean(ec**2)), name=name or rp_path.split('/')[-1], n=len(R), hz=rate, dt_p5=np.percentile(dt,5), dt_p95=np.percentile(dt,95),
        reuse=reuse*100, lat_med=np.median(lat), lat_p95=np.percentile(lat,95), cpu=np.nanmedian(R[:,6])*1e3,
        car_med=np.median(dcar)*100, car_p95=np.percentile(dcar,95)*100,
        ratio=np.median(ch[m]/wd[m]) if m.sum() else np.nan, still_med=np.median(stp[st])*100, still_p95=np.percentile(stp[st],95)*100,
        dev_med=np.median(dev[mv])*100, dev_p95=np.percentile(dev[mv],95)*100, cxx=np.median(R[:,7]))
def show(rows):
    print(f"{'kosu':28s} {'Hz':>5s} {'dt p5-p95 ms':>13s} {'tekrar%':>7s} {'gecikme med/p95':>16s} {'CPU ms':>6s} {'aracPF fark med/p95 cm':>22s} {'yol/teker':>9s} {'durgun adim cm':>15s} {'hareket adim hatasi cm':>22s} {'ZED-tutarlilik yolboyu RMS/med cm':>34s} {'yanal RMS':>9s}")
    for d in rows:
        print(f"{d['name']:28s} {d['hz']:5.1f} {d['dt_p5']:6.0f}-{d['dt_p95']:<6.0f} {d['reuse']:7.1f} {d['lat_med']:7.0f}/{d['lat_p95']:<8.0f} {d['cpu']:6.1f} {d['car_med']:10.1f}/{d['car_p95']:<11.1f} {d['ratio']:9.3f} {d['still_med']:6.1f}/{d['still_p95']:<8.1f} {d['dev_med']:10.1f}/{d['dev_p95']:<11.1f} {d['cons_a']:18.1f}/{d['cons_a_med']:<15.1f} {d['cons_c']:9.1f}")
if __name__=='__main__':
    npz=sys.argv[1]; t1=float(sys.argv[2]); show([metrics(p,npz,t1) for p in sys.argv[3:]])
