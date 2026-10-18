# RC Autonomous Race Car — ROS 2 Workspace

> **Fork of [memre12/RC_Car_Codes](https://github.com/memre12/RC_Car_Codes)** by
> Mehmet Emre Demir ([@memre12](https://github.com/memre12)), who built the
> original stack (drivers, particle filter setup, path/speed references,
> lattice planner, pure pursuit and the MPC tracking controller).
> This fork adds a **delay-compensated EKF localization** (`rc_localization`)
> and its integration with the MPC — see [What this fork adds](#what-this-fork-adds).

Autonomous driving stack for a 1/10-scale RC race car running on an
**NVIDIA Jetson Xavier (16 GB)** with a YDLidar, a VESC motor controller
and a particle-filter localization against a pre-built map.

Targets **ROS 2 Foxy** on the car; also builds on Humble (the tf2 include
deprecation warning on Humble is expected and harmless).

## Demo video

[![Watch the RC autonomous race car demo](https://img.youtube.com/vi/IBQtza92N10/maxresdefault.jpg)](https://www.youtube.com/watch?v=IBQtza92N10)

Click the preview to watch the video on YouTube.

## System architecture

![RC Autonomous Race Car system architecture](docs/architecture/system_architecture.png)

Editable source: [`system_architecture.drawio`](docs/architecture/system_architecture.drawio)

- **path_publisher / desired_speed_pub** (`v_ref_gen`) — publish the global
  centerline (`/path`, latched) and the speed reference (`/target_speed`,
  **m/s**) computed from path curvature/lateral error.
- **scan_to_grid / mission_planner / lattice_planner** (`lattice_planner_pkg`)
  — local obstacle avoidance. The mission planner watches a corridor around
  the centerline; on obstruction the lattice planner generates laterally
  offset spline candidates and `/selected_path` deviates around the
  obstacle, otherwise the centerline is forwarded. See
  `lattice_planner_pkg/README.md`.
- **race_car_controller** (`rc_pure_pursuit`) — pure pursuit tracking of
  `/selected_path` with speed-adaptive lookahead, expressed in the real
  vehicle parameters (wheelbase 0.28 m, max steer 29.85°). See
  `rc_pure_pursuit/race_car_controller/README.md`.
- **particle_filter**, **slam_toolbox** — localization / mapping.
- **vesc**, **ydlidar_ros2_driver** — hardware drivers.
- **race_car_launcher** — top-level launch files.

## What this fork adds

Branch `feature/ekf-localization` — Mehmet Görkem Keskin.

The particle filter pose arrives ~0.2 s late and at ~8 Hz. The new
**`rc_localization`** package fuses it with the ZED2 IMU and the VESC speed in an
EKF with Autoware-style delay compensation and publishes a **50 Hz pose predicted
to the present** on `/localization/odom` (same `nav_msgs/Odometry` interface).
Details, parameters and results: [`src/rc_localization/README.md`](src/rc_localization/README.md).

| Change | Where |
|---|---|
| EKF, twist estimator, monitors (`localization_monitor.py`, `live_status.py`), offline replay, 11 unit tests | `src/rc_localization/` (new) |
| `localization:=pf\|ekf` (default `pf`), `v_max` / `a_lat_max` overrides | `mpc_tracking_controller/launch/mpc.launch.py` |
| Optional EKF mode for the particle filter (`ekf:=true`, defaults unchanged, not yet tested on the car) | `particle_filter/` |
| One-command Gazebo test: path + speed ref + EKF + MPC | `rc_localization/launch/sim_test.launch.py` |

Results (Gazebo car-like mode, error against the true pose):

| | PF as seen by the controller | EKF |
|---|---|---|
| Rate | 8.4 Hz | 50 Hz |
| Pose age | ~266 ms | ~10 ms |
| Position error mean / max | 26 / 54 cm | 7 / 19 cm |

On 8 recorded car bags: max jump 5.6 cm, no resets. Builds and passes the tests on
Foxy (`ros:foxy` Docker) and Humble. **Not yet driven on the real car.**

```bash
# car: EKF next to the existing stack, then the MPC on the EKF pose
ros2 launch rc_localization ekf_localization.launch.py
ros2 launch mpc_tracking_controller mpc.launch.py localization:=ekf
```

Measured on the bags: **1 m/s ≈ 2080 ERPM** at the VESC (`/sensors/core`); the
computed 3365.4 below does not match the recorded data (the MPC uses 2171).

## Units convention

**All speeds in the stack are m/s.** ERPM appears only at the VESC
boundary; the controller converts using the measured drivetrain
(`race_car_parameters.txt`):

```
ERPM = v · 60 · (motor_poles/2) · gear_ratio / (2π · wheel_radius)
     = v · 3365.4        (14 poles, gear 2.769, wheel r = 0.055 m)
```

Note: VESC ERPM is electrical RPM = mechanical RPM × pole **pairs**.
Reference points: 2000 ERPM ≈ 0.6 m/s (slow walk), 3000 ≈ 0.9, 6000 ≈ 1.8.

## Build (Jetson Xavier)

```bash
cd ~/racecar_ws
sudo apt update && rosdep update rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Notes:
- `lattice_planner_pkg` and `race_car_controller` default to Release even
  without the flag; passing it makes every package optimized. An
  unoptimized build is the most common cause of the stack feeling slow.
- If RAM pressure kills the compiler on the Jetson, limit parallelism:
  `colcon build --parallel-workers 2 --executor sequential`.

## Run

```bash
# 1. Hardware (lidar + VESC + joystick)
ros2 launch race_car_launcher drivers.launch.py

# 2. Autonomy (localization + path/speed refs + controller + Obstacle Avoidance)
ros2 launch race_car_launcher autonomous.launch.py
```

Joystick button 0 toggles autonomous mode (0.5 s debounce). While off, the
controller streams neutral servo / zero speed.

## Simulation

The autonomy stack can also be integrated with
[`RC_Car_Sim`](https://github.com/memre12/RC_Car_Sim). In simulation, do not
start the physical sensor drivers; sensor data is supplied by the simulator.
The planning and control components can be used with the simulated topics.

Build and start the simulator from its own workspace:

```bash
cd ~/RC_Car_Sim
./build.sh
source install/setup.bash
ros2 launch sim_pkg simulation_environment.launch.py
```

EKF + MPC test (RC_Car_Sim branch `car-like-mode`, see the
[fork](https://github.com/maximgorkim/RC_Car_Sim/tree/car-like-mode) for `clock_rate`):

```bash
# container
ros2 launch sim_pkg simulation_environment.launch.py vehicle:=car clock_rate:=1000
# host
ros2 launch rc_localization sim_test.launch.py                 # localization:=pf to compare
ros2 run rc_localization live_status.py --ros-args -p use_sim_time:=true
```

## Tuning quick reference

| What | Where | Note |
|---|---|---|
| Steering feel | `steering_gain` in `rc_pure_pursuit/.../config/params.yaml` | 2.35 = tuned-on-track response; 1.0 = pure geometric pure pursuit |
| Speed range | `min/max_speed` in `v_ref_gen/path_publisher/config/params.yaml` | m/s; PD output effectively spans [min, (min+max)/2] |
| Lookahead | `lookahead_distance`, `min/max_look_ahead_distance` (controller) | Ld scales with commanded speed |
| Obstacle sensitivity | `obstacle_threshold`, `path_width` (mission_planner) | corridor checked ahead of the car |
| Avoidance offsets | `candidate_offsets` (lattice_planner) | lateral spline goals, meters |
| Obstacle inflation | `inflation_radius` (scan_to_grid) | keep ≥ half vehicle width (0.158 m) |

Vehicle measurements live in `race_car_parameters.txt`.
