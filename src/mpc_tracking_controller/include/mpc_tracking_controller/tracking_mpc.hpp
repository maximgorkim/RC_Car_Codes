// Path-tracking MPC for the RC car (no ROS in this file).
//
// Model: kinematic bicycle (rear axle, wheelbase L), tracked point = the localisation
// point a metres ahead of the rear axle, in Frenet coordinates of the path:
//
//   psi_dot = v tan(delta) / L
//   e_y_dot = v sin(e_psi) + a psi_dot cos(e_psi)
//   s_dot   = (v cos(e_psi) - a psi_dot sin(e_psi)) / (1 - kappa(s) e_y)
//   e_psi_dot = psi_dot - kappa(s) s_dot
//   v_dot   = (g v_cmd(t - d_v) - v) / tau_v              (speed: dead time + first-order lag)
//   delta_dot = (delta_cmd(t - d_d) - delta) / tau_d      (steering, effective wheel angle)
//
// x = [e_y, e_psi, v, delta], u = [v_cmd, delta_eff_cmd]. The steering dead time is removed
// by starting the horizon when the new command takes effect; the speed channel's EXTRA dead
// time is an integer input delay of the model. The steering map (servo command -> effective
// angle, identified) is applied outside: the MPC optimises the effective angle.
//
// Optimisation: real-time iteration (linearise around the shifted previous solution,
// condensed QP in the absolute inputs), box + rate constraints, lateral-acceleration speed
// cap, Bryson weights; DenseQp (ADMM) with warm start.
#pragma once

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

#include "mpc_tracking_controller/dense_qp.hpp"
#include "mpc_tracking_controller/path_geometry.hpp"

namespace mpc_tc {

struct ModelParams {
  double wheelbase = 0.28;   // m (effective, in pose units)
  double lever_arm = 0.17;   // m, localisation point ahead of the rear axle
  double steer_tau = 0.04;   // s
  double speed_tau = 0.25;   // s
  double speed_gain = 1.005;
  // understeer: yaw rate = v tan(delta) / (L (1 + understeer v^2)) [s^2/m^2]; 0 = kinematic
  double understeer = 0.0;
};

struct MpcSettings {
  int horizon = 20;
  double dt = 0.1;           // s per stage
  int substeps = 4;          // integration substeps per stage
  int speed_delay_steps = 1; // extra dead time of the speed channel, in stages
  // Bryson weights (1 / acceptable value^2), as the IRL controller's running cost
  double max_e_lat = 0.10, max_e_psi = 0.15, max_e_v = 0.25, max_v_rate = 1.5, max_delta_rate = 1.0;
  double terminal_factor = 5.0;
  // constraints (delta in effective radians)
  double v_min = 0.0, v_max = 2.0, delta_lo = -0.314, delta_hi = 0.347;
  double v_rate_max = 2.0, delta_rate_max = 1.5, a_lat_max = 2.0;
  int cap_preview_stages = 6;  // lateral-acceleration cap over this many stages ahead (0.6 s, as IRL)
  // Friction circle: total acceleration the tyres give [m/s^2], 0 = off (separate limits, as before).
  // The speed change per stage is then also bounded by drive_axle_load * sqrt(grip^2 - a_lat^2),
  // a_lat from the previous solution where that speed change acts: a rear-driven car accelerating
  // out of a corner on a slippery floor otherwise spends the rear grip on traction and spins.
  double grip_accel = 0.0;
  double drive_axle_load = 0.5;
  int sqp_iterations = 1;      // 1 = real-time iteration
  QpSettings qp;
};

struct MpcResult {
  bool ok = false;
  double v_cmd = 0.0, delta_eff = 0.0;
  int qp_iterations = 0;
  bool qp_converged = false;
  std::vector<Eigen::Vector4d> x_pred;  // predicted states x_0..x_N (last linearisation)
  std::vector<double> s_pred;
};

class TrackingMpc {
 public:
  TrackingMpc(const ModelParams& m, const MpcSettings& s);
  // x0 at the time the new steering command takes effect; s0 its arc length;
  // (v_prev, d_prev): last commands sent (d_prev in effective radians)
  MpcResult solve(const Eigen::Vector4d& x0, double s0, double v_ref, const PathGeometry& path, double v_prev,
                  double d_prev);
  void reset() { have_nominal_ = false; }
  // one stage of the model; returns x_next, adds the travelled arc length to ds
  // yaw_dist: yaw-rate disturbance [rad/s] added to the model (learned along the path)
  Eigen::Vector4d step(const Eigen::Vector4d& x, double v_cmd, double d_cmd, double kappa, double& ds,
                       double yaw_dist = 0.0) const;
  const MpcSettings& settings() const { return set_; }
  // limits that change at run time (grip adaptation); the configured ones are the upper bounds
  void set_understeer(double k) { mod_.understeer = k; }
  // yaw-rate disturbance along the path (arc length -> rad/s); empty = none
  void set_yaw_disturbance(std::function<double(double)> d) { yaw_dist_ = std::move(d); }
  void set_grip_limits(double a_lat_max, double v_rate_max, double grip_accel) {
    set_.a_lat_max = a_lat_max;
    set_.v_rate_max = v_rate_max;
    set_.grip_accel = grip_accel;
  }

 private:
  void rollout(const Eigen::Vector4d& x0, double s0, const PathGeometry& path, double v_prev);
  ModelParams mod_;
  MpcSettings set_;
  std::function<double(double)> yaw_dist_;
  DenseQp qp_;
  int N_, n_, m_;
  Eigen::VectorXd u_nom_;                // [v_0, d_0, v_1, d_1, ...]
  bool have_nominal_ = false;
  std::vector<Eigen::Vector4d> xn_;      // nominal states
  std::vector<double> sn_, kn_, dn_;     // nominal arc length, curvature, yaw disturbance per stage
  Eigen::MatrixXd P_, A_;
  Eigen::VectorXd q_, l_, u_;
  std::vector<Eigen::Matrix<double, 4, Eigen::Dynamic>> S_;
};

// ---------------------------------------------------------------- state estimation
// Dead time + first-order lag driven by piecewise-constant commands, evaluated exactly.
class LaggedActuator {
 public:
  void configure(double dead, double tau, double gain) { dead_ = dead; tau_ = tau; gain_ = gain; }
  void reset(double t, double x, double u) { t_ = t; x_ = x; u_ = u; pending_.clear(); init_ = true; }
  bool initialised() const { return init_; }
  void command(double t_sent, double u) { pending_.emplace_back(t_sent + dead_, u); }
  double value_at(double t) const;  // does not change the state
  void advance(double t);           // commits the state up to t (t must not decrease)
  double last_command() const { return pending_.empty() ? u_ : pending_.back().second; }

 private:
  double dead_ = 0.0, tau_ = 0.05, gain_ = 1.0;
  double t_ = 0.0, x_ = 0.0, u_ = 0.0;
  bool init_ = false;
  std::deque<std::pair<double, double>> pending_;  // (effective time, command)
};

struct ControllerParams {
  ModelParams model;
  MpcSettings mpc;
  double steer_dead_time = 0.05;   // s
  double speed_dead_time = 0.18;   // s
  std::vector<double> steer_map_cmd{-0.32, -0.2, -0.1, 0.0, 0.1, 0.2, 0.32};
  std::vector<double> steer_map_eff{-0.314, -0.217, -0.092, -0.008, 0.089, 0.204, 0.347};
  bool use_steer_map = true;
  double pose_latency_position = 0.239;  // s, pose arrival - time the position refers to
  double pose_latency_heading = 0.200;   // s
  double heading_bias = 0.045;           // rad, course - localisation heading
  double compute_delay = 0.014;          // s, minimum pose-arrival -> command-publication delay assumed
  double delta_cmd_max = 0.32;           // rad, servo command bound
  double control_period = 0.1;           // s, rate limits per step as the IRL node
  double abort_e_lat = 0.8, abort_e_psi = 1.0;
  bool use_imu = true;                   // dead-reckon the latency window with the IMU yaw rate
  // Online steering-offset (servo trim) estimate: the IMU yaw rate vs the model's v tan(delta) / L
  // near straight driving, low-passed with this time constant [s]; 0 = off (no offset, as before).
  // The bags show 0.002-0.010 rad of residual trim that changes between sessions.
  double steer_offset_tau = 0.0;
  double steer_offset_max = 0.03;        // rad, |estimate| bound
  double steer_offset_min_speed = 0.5;   // m/s, below this the yaw rate says little about the angle
  double steer_offset_max_delta = 0.06;  // rad, only near straight driving (understeer, map errors)
  // Grip adaptation (wet / slippery floor). The kinematic model holds while the tyres grip; when
  // they saturate the measured yaw rate departs from v tan(delta) / L (the car plows or spins).
  // Then the available grip is what the car achieved (v r), and the MPC's lateral-acceleration
  // speed cap and acceleration limit shrink to grip_margin of it; without saturation the estimate
  // recovers towards grip_nominal with grip_recover_tau. Off by default.
  bool grip_adaptation = false;     // also turns the MPC's friction circle on
  double grip_nominal = 1.0;      // g, the floor the configured limits were made for
  double grip_min = 0.15;         // g
  double grip_margin = 0.8;       // use this fraction of the estimate
  double grip_recover_tau = 30.0;  // s (8 s: the car forgot a wet stretch before the next lap)
  double slip_rel = 0.15;         // saturation: |r_model - r_imu| > slip_rel |r_model| + slip_abs
  double slip_abs = 0.08;         // rad/s, above the model / IMU noise (validated RMSE 0.06-0.07)
  double slip_min_speed = 0.8;    // m/s
  // The slide test runs on every IMU sample: yaw rate and model both low-passed with slip_filter_tau,
  // a slide once the mismatch has lasted slip_min_time.
  double slip_filter_tau = 0.04;  // s
  double slip_min_time = 0.08;    // s
  // Leaving the slippery stretch: cornering at >= grip_probe_level of the estimate without a slide
  // is evidence that the grip is back, the estimate then recovers with grip_probe_tau (else with
  // grip_recover_tau). If it was not back the next slide pulls it down again within ~0.1 s.
  double grip_probe_level = 0.6;
  double grip_probe_tau = 5.0;    // s (stack_sim wet stretch: 2 s crashed 3-5/16 on re-entry, 5 s 0/16)
  // Online understeer estimate (model.understeer) from yaw rate vs steering while gripping, low-passed
  // with this time constant [s]; 0 = off. Saturation is then judged against the learned model, so
  // the tyres' normal (linear) understeer is not taken for a slide.
  double understeer_tau = 0.0;
  double understeer_max = 0.3;    // s^2/m^2
  // Yaw-rate residual map along the path (Pi-MPC idea, in arc length instead of time): the part of
  // the IMU yaw rate the model does not explain is learned per yaw_map_bin [m] (0 = off) while the
  // car grips, blended with yaw_map_gain per pass, and fed to the MPC model: whatever repeats lap
  // after lap (tyres, map, trim, the floor) is predicted instead of corrected after the fact.
  double yaw_map_bin = 0.0;
  double yaw_map_gain = 0.3;
  double yaw_map_max = 1.0;       // rad/s
};

enum class Status { kOk, kNoPath, kNoSpeed, kAbort };

struct ControlOutput {
  Status status = Status::kNoPath;
  double v_cmd = 0.0, delta_cmd = 0.0;  // delta_cmd: servo-command radians
  Projection meas;                      // projection of the pose as received
  Eigen::Vector4d x0 = Eigen::Vector4d::Zero();
  double s0 = 0.0, t_start = 0.0;
  MpcResult mpc;
};

class TrackingController {
 public:
  explicit TrackingController(const ControllerParams& p);
  void set_path(const std::vector<double>& x, const std::vector<double>& y);
  bool has_path() const { return path_ != nullptr; }
  const PathGeometry* path() const { return path_.get(); }
  void on_imu(double t, double yaw_rate);
  void on_speed(double t, double v);
  // pose_age: arrival - pose stamp when the stamp is the measurement time
  // (pose_latency_* then only hold the residual); 0 with now()-stamped poses
  ControlOutput on_pose(double t_arrival, double x, double y, double yaw, double v_ref, double pose_age = 0.0);
  // every command actually published (stop commands too); delta in servo-command radians
  void on_command_sent(double t, double v_cmd, double delta_cmd);
  void reset_mpc() { mpc_.reset(); }
  double steer_to_eff(double d_cmd) const;
  double eff_to_steer(double d_eff) const;
  double steer_offset() const { return steer_offset_; }
  double grip() const { return grip_; }  // g, current estimate (grip_nominal when off)
  double understeer() const { return understeer_; }
  bool slipping() const { return slipping_; }

 private:
  double sample(const std::deque<std::pair<double, double>>& buf, double t) const;
  ControllerParams p_;
  TrackingMpc mpc_;
  std::unique_ptr<PathGeometry> path_;
  std::deque<std::pair<double, double>> imu_, vel_;
  LaggedActuator steer_, speed_;
  double last_v_cmd_ = 0.0, last_d_cmd_ = 0.0;
  double compute_ema_ = 0.0;   // measured pose-arrival -> publication delay (exp. moving average)
  double pending_arrival_ = -1.0;
  double steer_offset_ = 0.0, last_offset_t_ = -1.0;
  void update_steer_offset(double t_arr);
  // mean IMU yaw rate over [t0, t1]; false if too few samples
  bool imu_mean(double t0, double t1, double& r) const;
  double grip_ = 1.0, last_grip_t_ = -1.0, grip_before_ = 1.0, episode_max_ = 0.0;
  double understeer_ = 0.0, last_s_ = -1.0;
  std::vector<double> yaw_map_;
  double yaw_map_at(double s) const;
  double rf_ = 0.0, rmf_ = 0.0, last_imu_t_ = -1.0, slip_time_ = 0.0;
  bool slipping_ = false;
  void detect_slip(double t, double r);
  double yaw_rate_of(double v, double delta) const;
  void update_grip(double t_arr);
};

}  // namespace mpc_tc
