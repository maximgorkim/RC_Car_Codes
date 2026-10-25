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
