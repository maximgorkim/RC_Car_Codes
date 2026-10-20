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
  Eigen::Vector4d step(const Eigen::Vector4d& x, double v_cmd, double d_cmd, double kappa, double& ds) const;
  const MpcSettings& settings() const { return set_; }

 private:
  void rollout(const Eigen::Vector4d& x0, double s0, const PathGeometry& path, double v_prev);
  ModelParams mod_;
  MpcSettings set_;
  DenseQp qp_;
  int N_, n_, m_;
  Eigen::VectorXd u_nom_;                // [v_0, d_0, v_1, d_1, ...]
  bool have_nominal_ = false;
  std::vector<Eigen::Vector4d> xn_;      // nominal states
  std::vector<double> sn_, kn_;          // nominal arc length, curvature per stage
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
};

}  // namespace mpc_tc
