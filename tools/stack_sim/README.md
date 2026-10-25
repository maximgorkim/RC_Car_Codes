# stack_sim — the whole stack in closed loop, without ROS

Plant (the identified car) → IMU / VESC / PF pose → `twist_estimator` + `ekf_localizer` logic
(the real `ekf_core`) → MPC (the real `mpc_core`, as `mpc_node`) — against ground truth.
Deterministic per seed, ~0.3 s per 80 s run. Not a colcon package (`COLCON_IGNORE`); nothing here
runs on the car.

```bash
cmake -S tools/stack_sim -B build/stack_sim -DCMAKE_BUILD_TYPE=Release && cmake --build build/stack_sim
P=src/v_ref_gen/path_publisher/maps/smooth_loop_path3_edit.csv
build/stack_sim/stack_sim $P ctrl=ekf pf=scan speed=1.5
python3 tools/stack_sim/compare.py $P --seeds 16 --common "speed=1.5" -- "ctrl=pf" "ctrl=ekf" "ctrl=ekf trim=0.01 mpc.steer_offset_tau=3"
```

Scenario keys (car errors the EKF / MPC should absorb) and the EKF / MPC parameter keys are
listed at the top of `stack_sim.cpp`. Output: true lateral error (all / straights / curves,
`elat_st_mean` = steady offset), steering activity, EKF position / heading error, jumps between
EKF outputs, rejected poses, resets, MPC aborts.

Defaults follow `RC_Car_Sim car_like.json`. Two caveats:
* `pf_pos_extra=0.04` (position older than heading) is from fair_irl_1; the 2026-10-05 bags fit 0.
* `smooth_loop_path3_edit.csv` is tighter than the car can steer at s ≈ 22–27 m (|κ| up to 2.1 1/m,
  the car reaches 1.2–1.3): the largest curve errors there are the path's, not the controller's.
