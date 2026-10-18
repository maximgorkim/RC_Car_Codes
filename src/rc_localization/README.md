# rc_localization — low-latency EKF localization

RC-car version of the full-size car's localization architecture
(`autoware_ekf_localizer` + `gyro_odometer`): a **fast** twist from the VESC
and the IMU predicts the motion, and the **slow, delayed** LiDAR pose from the
particle filter corrects the drift. The LiDAR pose is applied **at the time it
was measured** (delay compensation), so the output is current even though the
LiDAR pose is ~0.2 s old. Builds on Foxy and Humble.

```
 /sensors/core (VESC ERPM, 50 Hz) ─┐
 /zed2/zed_node/imu/data (190 Hz) ─┴─► twist_estimator ─► /localization/twist_with_cov ─┐
                                                                                       ├─► ekf_localizer ─► /localization/odom (50 Hz)
 /scan ─► particle_filter ─► /pf/pose/odom (8 Hz, ~0.2 s late) ────────────────────────┘                ─► TF odom → base_link
```

## Nodes

| Node | What it does |
|---|---|
| `twist_estimator` | Speed `v = ERPM / erpm_per_mps` from the VESC and yaw rate from the IMU gyro (not from the steering, which slips). One twist per IMU message, **stamped with the IMU time**. |
| `ekf_localizer` | EKF with delay compensation. Publishes `nav_msgs/Odometry` at a fixed rate, predicted to *now*, plus the TF. Same message as `/pf/pose/odom`, so consumers only need a topic change. |
| `localization_monitor.py` | Lag/quality monitor (port of the big car's `ndt_ekf_lag_monitor.py`): output rate, TF age, LiDAR pose age, EKF vs LiDAR residual, jumps; optional CSV log. |
| `live_status.py` | Live terminal dashboard while driving: speed / target / command, steering, path error, MPC solve time, EKF rate and health, and in Gazebo the EKF and PF error against the true pose. `ros2 run rc_localization live_status.py --ros-args -p use_sim_time:=true` |
| `sim_vesc_bridge.py` | Gazebo only: `/sim/vesc/erpm` (Float64) → `/sensors/core` (VescStateStamped), started by `mpc.launch.py use_sim_time:=true`. |
| `ekf_replay` | Offline tool: runs the same filter code on a recorded sensor stream in seconds. Used for tuning. |

## The filter

State (11): `x, y, yaw, yaw_bias, vx, wz, gyro_bias, speed_scale, vy, ay, ay_bias`
(the last three only with `model.lateral_dynamics`)

* **Prediction** (rear-axle kinematics, base_link `lever_arm` ahead of the rear axle):
  `x += (vx cos(yaw+yaw_bias) − a·wz sin(yaw+yaw_bias))·dt`,
  `y += (vx sin(yaw+yaw_bias) + a·wz cos(yaw+yaw_bias))·dt`,
  `yaw += wz·dt`; everything else is a random walk.
* **Twist update**: `z = [speed_scale·vx, wz + gyro_bias]`. The filter learns the
  gyro bias and the ERPM→m/s scale error on its own.
* **Pose update**: `z = [x, y, yaw]`, with
  * **Mahalanobis gate** (χ²₃ 99.9 % = 16.27): bad matches are rejected;
    after `max_consecutive_rejections` in a row the filter restarts from the LiDAR.
  * **Time uncertainty**: `R += σ_t²·u·uᵀ`, `u` = velocity. A pose whose stamp
    is off by δt is off by `v·δt` along the motion. Needed because today's
    particle filter stamps with `now()` and its latency jitters.
* **Delay compensation** (`DelayedEkf`): every measurement and the state after
  it are kept for `history_length`. A late pose is inserted at its own time and
  all later measurements are replayed (~40 for a 0.2 s delay; microseconds).
* `yaw_bias` is Autoware's idea: the direction of travel differs from the
  LiDAR heading (slip / mounting offset).
* **Side slip** (`model.lateral_dynamics: true`, launch `lateral_dynamics:=true`):
  the lateral velocity `vy` of base_link becomes a state driven by the IMU
  lateral acceleration, `vy' = vy + (ay − vx·wz − (vy − a·wz)/slip_tau)·dt`,
  measurement `ay_meas/accel_scale = ay + ay_bias`. A slide shows up as
  `ay ≠ vx·wz`. `slip_tau` (2 s) only bounds accelerometer drift; the slide
  itself is in the measured `ay` (0.3 s halves the slide tracking).
  ZED calibration from the bags: `accel y = 1.24·v·wz − 0.37` (every bag 1.13–1.29)
  → `accel_scale: 1.24`, `init.ay_bias: −0.30`.
  Unit test (car sliding 1.2 m/s at 7 m/s): error during slides 0.47 → 0.05 m.
  Gazebo at 6 m/s: EKF error p95 80 → 22 cm, resets 5 → 1. Bags (≤ 2.5 m/s, no
  slides): rejected poses 18 → 12, but max jump 5.6 → 11.7 cm, hence off by default.

The core (`ekf_core.hpp/.cpp`) has no ROS dependency and is covered by unit
tests (`test/test_ekf_core.cpp`): numeric Jacobian check, out-of-order equals
in-order, too-old measurements, and a closed-loop 10 m/s simulation with
gyro bias, speed-scale error, 0.2 s pose delay and 2 m outliers.

## Run

```bash
colcon build --packages-up-to rc_localization
# particle filter as today, then:
ros2 launch rc_localization ekf_localization.launch.py              # + monitor:=true
```

* **TF**: only one node may publish `odom → base_link`. Either turn the
  particle filter's TF off, or run `publish_tf:=false` to compare side by side.
* **Consumers**: set `odom_topic: /localization/odom` in
  `race_car_controller`, `mpc_tracking_controller`, `lattice_planner_node`,
  `mission_planner`, `desired_speed_pub`. No code change.
* **Initial pose**: the first LiDAR pose, or RViz *2D Pose Estimate* (`/initialpose`).

### In Gazebo (RC_Car_Sim, branch car-like-mode)

```bash
# container: ros2 launch sim_pkg simulation_environment.launch.py vehicle:=car clock_rate:=1000
ros2 launch rc_localization sim_test.launch.py      # path + speed ref + EKF + MPC on sim time
ros2 launch rc_localization sim_test.launch.py target_speed:=3.0 a_lat_max:=4.0   # constant speed; v_max follows it
```
`clock_rate:=1000` matters: with Gazebo's default 10 Hz /clock every sensor
stamp is quantized to 0.1 s. Args: `localization:=pf|ekf`, `target_speed:=<m/s>`,
`lateral_dynamics:=true|false`.

### Robustness

Non-finite measurements are dropped; if the state becomes non-finite or the
position variance exceeds `max_position_variance` (4 m²), nothing is published
and the filter restarts from the next LiDAR pose; after
`max_consecutive_rejections` rejected LiDAR poses it restarts from the LiDAR.
Builds and passes its tests on Foxy (`ros:foxy-ros-base`) and Humble.

### On a bag

```bash
ros2 launch rc_localization ekf_localization.launch.py use_sim_time:=true monitor:=true csv_path:=/tmp/ekf.csv
ros2 bag play <bag> --clock 200      # --clock rate >= output_rate, otherwise the 50 Hz timer is quantized
```

### Offline tuning

`ekf_replay events.csv key=value ... [--out traj.csv] [--innov innov.csv]`
(see the header of `tools/ekf_replay.cpp` for the CSV format and keys).

## Parameters that matter

| Parameter | Value | Where it comes from |
|---|---|---|
| `erpm_per_mps` | 2080 | Measured on the bags (VESC ERPM vs PF speed), then refined by the EKF's `speed_scale` (1.015). **The 3365.4 in `race_car_parameters.txt` is ~1.6× too high.** |
| `pose_stamp_offset` | 0.20 s | PF latency: gyro vs PF yaw-rate cross-correlation 160–220 ms; offset sweep on 4 bags has its minimum at 0.20. Set to 0 once the PF stamps with the scan time. |
| `pose_time_stddev` | 0.04 s | PF latency jitter; cut rejected poses ~1 % → ~0.5 %. ~0.01 after the PF fix. |
| `model.lever_arm` | 0.17 m | Sweep 0–0.35 m on 8 bags: best 0.17 (= MPC identification). |
| `init.yaw_bias` | 0.05 rad | Learned 0.040–0.057 on the bags (MPC identification: 0.045). |
| `process_noise.pos` | 0.02 | Larger values follow the PF noise (jumps up to 9 cm). |

## Results on the 2026-10-05 bags

Recorded 2026-10-05 (no `/scan`, so the PF output in the bag is the LiDAR
pose). 8 usable bags, ~5.5 min, up to ~2.1 m/s (measured scale).

| | Old: `/pf/pose/odom` | New: `/localization/odom` | Goal |
|---|---|---|---|
| Output rate | 8.4 Hz (irregular, 120 ms median gap) | **50 Hz** fixed | ≥ 50 Hz |
| Pose age when received | ~205 ms | **0 ms** (predicted to now) | |
| Pose age at a random time (rate + latency) | mean **266 ms**, p95 330, max ~400 | mean **10 ms**, max 20 | < 20–25 ms |
| → position lag at 10 m/s | **2.7 m** mean, 4 m max | **0.1 m** mean, 0.2 m max | |
| Jumps between outputs | PF jitter ~2.5 cm per pose | max **2.0 cm** (ROS run), 4.7 cm (worst bag, offline) | < 10 cm |
| Rejected LiDAR poses | – | 18 of 3178 (0.6 %), 0 resets | |

* Innovation (LiDAR pose vs EKF estimate at the pose time, before the update):
  RMS 7–9 cm, almost all along the track (6–8 cm) and growing with speed,
  zero mean: the PF latency jitter. Cross-track 3–4 cm, yaw 1.3–1.8°.
* Learned on every bag: `yaw_bias` 0.041–0.057 rad, `gyro_bias` within
  ±0.002 rad/s, `speed_scale` 0.98–1.005 (with 2080 ERPM per m/s).
* ROS run (`mpc13metersc_fix`, `ros2 bag play --clock 200`): 50.0 Hz, odom
  age 0.0 ms, max jump 2.0 cm, 1 rejected pose in 33 s.
* CPU: the filter itself (`ekf_replay`, all delay-compensation replays
  included) needs 0.05 s of CPU for the 66 s `notsafeirl1` bag on a Ryzen 5
  3600X, i.e. < 0.1 % of one core. ROS message overhead comes on top; not yet
  measured on the Xavier.

## Known limits / next steps

1. **Particle filter**: done but not run yet (no `range_libc` off the car):
   `ros2 launch particle_filter localize_launch.py ekf:=true` (no TF, one
   update per scan stamped with the scan time, twist motion model, `viz: 0`).
   Then EKF `pose_stamp_offset:=0.0` (+ `publish_tf` on), and check / tune
   `pose_time_stddev` (~0.01) on the car.
2. **Scan deskew**: at 10 m/s the car moves 1 m during one 10 Hz scan.
3. **TF tree**: move to `map → odom → base_link` (the path and the planners
   hard-code `odom` today; `world_frame` is a parameter here).
4. The bags reach ~2.5 m/s only; verify at higher speed step by step
   (VESC `speed_max` 23250 ERPM ≈ 11 m/s with the measured 2080 ERPM per m/s).
