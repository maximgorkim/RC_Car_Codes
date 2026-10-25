# synth — synthetic world + sensor bag with ground truth

The car has no ground truth, so the particle filter can only be judged indirectly on real bags.
Here a map is generated around a path, the true trajectory comes from `tools/stack_sim`
(`--series`), and a bag with the car's topics (G4 LiDAR with per-beam timing, ZED VO with scale
error, IMU, VESC, twist) is written; the REAL particle filter runs on it (`pf_replay.py`) and is
compared with `/truth/odom`.

```bash
S=/tmp/synth; P=src/v_ref_gen/path_publisher/maps/smooth_loop_path3_edit.csv
build/stack_sim/stack_sim $P speed=2.5 mpc.v_max=2.5 mpc.a_lat_max=3 laps=2 --series $S/run.csv
python3 tools/synth/make_world.py $P $S/world/box            # corridor + boxes (--boxes 0: bare)
python3 tools/synth/make_bag.py $S/run.csv $S/world/box.yaml $S/bag
T=src/particle_filter/tools
python3 $T/pf_replay.py $S/bag $S/a.npz --map $S/world/box.yaml --set scan_synchronous:=true
python3 $T/pf_replay.py $S/bag $S/b.npz --map $S/world/box.yaml --set scan_synchronous:=true \
    motion_noise:=proportional motion_source:=twist max_particles:=2000 scan_deskew:=true
python3 tools/synth/eval_pf.py $S/bag $S/a.npz $S/b.npz
```

Results (2026-10-25, 2.5 m/s, PC CPU ray marching): B mode 4.0 cm / 1.16 deg -> twist + proportional
+ 2000 particles + deskew 1.6 cm / 0.21 deg at half the CPU. More beams (angle_step 4) alone: no gain.
The synthetic LiDAR stamps the first beam and orders beams by angle in time; whether the real G4
does the same must be checked (`particle_filter/tools/deskew_check.py`) before `deskew:=true`.

## slam_toolbox (slam/)
`slam/slam_run.sh BAG OUT` runs the car's mapping chain (`rc_localization/launch/mapping.launch.py`: EKF dead
reckoning + `scan_for_slam.py` + slam_toolbox sync) on a synthetic bag; `slam/slam_eval.py OUT...` compares
the SLAM pose with the truth after a rigid alignment (the SLAM map has its own frame). PF on the SLAM map:
`pf_replay.py ... --map OUT/map.yaml --init=x,y,yaw` (start pose in the SLAM frame), `eval_pf.py ... --frame x,y,yaw`.

Results (2026-10-25): with the driver's 0.0 invalid beams the map falls apart (6.5 m mean error); with +inf,
1 m/s: 10-13 cm, 2 m/s: 15 cm (p95 36). PF (best mode) on the 1 m/s SLAM map: 4.2 cm over the run, 2.3 cm
within 5 s windows (true map 1.1 cm). Resolution 0.025, minimum_travel 0.1, async, deskewed scans: no clear gain.
