#include "mpc_tracking_controller/tracking_mpc.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace mpc_tc {

// ======================================================================= model
TrackingMpc::TrackingMpc(const ModelParams& m, const MpcSettings& s) : mod_(m), set_(s), qp_(s.qp) {
  N_ = set_.horizon;
  n_ = 2 * N_;
  m_ = 4 * N_;
  u_nom_ = Eigen::VectorXd::Zero(n_);
  xn_.resize(N_ + 1);
  sn_.resize(N_ + 1);
  kn_.resize(N_ + 1);
  S_.assign(N_ + 1, Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4, n_));
  P_.resize(n_, n_);
  q_.resize(n_);
  A_ = Eigen::MatrixXd::Zero(m_, n_);
  l_.resize(m_);
  u_.resize(m_);
  // constraint matrix is constant: [I; D], D z = [u_0; u_1 - u_0; ...]
  A_.topRows(n_).setIdentity();
  for (int k = 0; k < N_; ++k)
    for (int j = 0; j < 2; ++j) {
      A_(n_ + 2 * k + j, 2 * k + j) = 1.0;
      if (k > 0) A_(n_ + 2 * k + j, 2 * (k - 1) + j) = -1.0;
    }
  qp_.resize(n_, m_);
}

Eigen::Vector4d TrackingMpc::step(const Eigen::Vector4d& x, double v_cmd, double d_cmd, double kappa,
                                  double& ds) const {
  const int M = std::max(1, set_.substeps);
  const double h = set_.dt / M;
  const double av = std::exp(-h / mod_.speed_tau), ad = std::exp(-h / mod_.steer_tau);
  const double vt = mod_.speed_gain * v_cmd;
  const double L = mod_.wheelbase, a = mod_.lever_arm;
  Eigen::Vector4d s = x;
  for (int i = 0; i < M; ++i) {
    // actuators exactly (piecewise-constant input), kinematics by the midpoint rule
    const double v1 = vt + (s[2] - vt) * av, d1 = d_cmd + (s[3] - d_cmd) * ad;
    const double vm = 0.5 * (s[2] + v1), dm = 0.5 * (s[3] + d1);
    const double r = vm * std::tan(dm) / (L * (1.0 + mod_.understeer * vm * vm));
    auto f = [&](double ey, double ep, double& dey, double& dep, double& dsd) {
      const double den = std::max(1.0 - kappa * ey, 0.1);
      dsd = (vm * std::cos(ep) - a * r * std::sin(ep)) / den;
      dey = vm * std::sin(ep) + a * r * std::cos(ep);
      dep = r - kappa * dsd;
    };
    double k1y, k1p, k1s, k2y, k2p, k2s;
    f(s[0], s[1], k1y, k1p, k1s);
    f(s[0] + 0.5 * h * k1y, s[1] + 0.5 * h * k1p, k2y, k2p, k2s);
    s[0] += h * k2y;
    s[1] += h * k2p;
    ds += h * k2s;
    s[2] = v1;
    s[3] = d1;
  }
  return s;
}

void TrackingMpc::rollout(const Eigen::Vector4d& x0, double s0, const PathGeometry& path, double v_prev) {
  xn_[0] = x0;
  sn_[0] = s0;
  const int nd = set_.speed_delay_steps;
  for (int k = 0; k < N_; ++k) {
    kn_[k] = path.kappa_at(sn_[k]);
    const double vc = (k - nd >= 0) ? u_nom_[2 * (k - nd)] : v_prev;
    double ds = 0.0;
    xn_[k + 1] = step(xn_[k], vc, u_nom_[2 * k + 1], kn_[k], ds);
    sn_[k + 1] = sn_[k] + ds;
  }
  kn_[N_] = path.kappa_at(sn_[N_]);
}

MpcResult TrackingMpc::solve(const Eigen::Vector4d& x0, double s0, double v_ref, const PathGeometry& path,
                             double v_prev, double d_prev) {
  const int nd = set_.speed_delay_steps;
  const double dt = set_.dt;
  // lateral-acceleration cap per stage, on the curvature where the command acts
  // With the friction circle on, the cap also respects the braking distance over the whole horizon
  // (backward pass on the nominal arc lengths): the rear-axle motor brake gives only
  // drive_axle_load * grip, so on a slippery floor the 0.6 s preview is far too short.
  std::vector<double> brake_cap;
  auto cap_at = [&](int k) {
    double kmax = 1e-6;
    for (int j = k + nd; j <= std::min(N_, k + nd + set_.cap_preview_stages); ++j)
      kmax = std::max(kmax, std::fabs(kn_[std::min(j, N_)]));
    const double c = std::sqrt(set_.a_lat_max / kmax);
    return brake_cap.empty() ? c : std::min(c, brake_cap[std::min(k + nd, N_)]);
  };
  auto update_brake_cap = [&]() {
    if (set_.grip_accel <= 0.0) return;
    brake_cap.assign(N_ + 1, 0.0);
    const double a_b = set_.drive_axle_load * set_.grip_accel;
    brake_cap[N_] = std::sqrt(set_.a_lat_max / std::max(std::fabs(kn_[N_]), 1e-6));
    for (int k = N_ - 1; k >= 0; --k) {
      const double ds = std::max(sn_[k + 1] - sn_[k], 0.0);
      brake_cap[k] = std::min(std::sqrt(set_.a_lat_max / std::max(std::fabs(kn_[k]), 1e-6)),
                              std::sqrt(brake_cap[k + 1] * brake_cap[k + 1] + 2.0 * a_b * ds));
    }
  };
  if (!have_nominal_) {
    // cold start: the reference speed and the kinematic feed-forward steering along the path
    double s = s0;
    for (int k = 0; k < N_; ++k) {
      u_nom_[2 * k] = std::min(v_ref, set_.v_max);
      u_nom_[2 * k + 1] = std::min(set_.delta_hi, std::max(set_.delta_lo, std::atan(mod_.wheelbase * path.kappa_at(s))));
      s += std::max(v_ref, 0.3) * dt;
    }
    qp_.mutable_solution() = u_nom_;
    qp_.reset_duals();
    have_nominal_ = true;
  }
  const Eigen::Vector4d q_st(dt / (set_.max_e_lat * set_.max_e_lat), dt / (set_.max_e_psi * set_.max_e_psi),
                             dt / (set_.max_e_v * set_.max_e_v), 0.0);
  const Eigen::Vector2d r_rate(dt / (set_.max_v_rate * set_.max_v_rate * dt * dt),
                               dt / (set_.max_delta_rate * set_.max_delta_rate * dt * dt));
  MpcResult out;
  for (int it = 0; it < std::max(1, set_.sqp_iterations); ++it) {
    rollout(x0, s0, path, v_prev);
    update_brake_cap();
    // --- condensed sensitivities S_k = d x_k / d z (finite differences of the stage map)
    for (auto& S : S_) S.setZero();
    for (int k = 0; k < N_; ++k) {
      const double vc = (k - nd >= 0) ? u_nom_[2 * (k - nd)] : v_prev;
      const double dc = u_nom_[2 * k + 1];
      Eigen::Matrix4d Ak;
      const double ex = 1e-6;
      for (int j = 0; j < 4; ++j) {
        Eigen::Vector4d xp = xn_[k], xm = xn_[k];
        xp[j] += ex;
        xm[j] -= ex;
        double d1 = 0.0, d2 = 0.0;
        Ak.col(j) = (step(xp, vc, dc, kn_[k], d1) - step(xm, vc, dc, kn_[k], d2)) / (2.0 * ex);
      }
      double d1 = 0.0, d2 = 0.0;
      const Eigen::Vector4d Bd = (step(xn_[k], vc, dc + ex, kn_[k], d1) - step(xn_[k], vc, dc - ex, kn_[k], d2)) / (2.0 * ex);
      S_[k + 1].noalias() = Ak * S_[k];
      S_[k + 1].col(2 * k + 1) += Bd;
      if (k - nd >= 0) {
        const Eigen::Vector4d Bv = (step(xn_[k], vc + ex, dc, kn_[k], d1) - step(xn_[k], vc - ex, dc, kn_[k], d2)) / (2.0 * ex);
        S_[k + 1].col(2 * (k - nd)) += Bv;
      }
    }
    // --- cost: sum_k (x_k - r_k)' Q (x_k - r_k) + rate terms; x_k = xn_k + S_k (z - u_nom)
    P_.setZero();
    q_.setZero();
    for (int k = 1; k <= N_; ++k) {
      const double w = (k == N_) ? set_.terminal_factor : 1.0;
      Eigen::Vector4d ref(0.0, 0.0, std::min(v_ref, cap_at(std::max(k - 1, 0))), 0.0);
      const Eigen::Vector4d d = xn_[k] - ref - S_[k] * u_nom_;
      const Eigen::Matrix<double, 4, Eigen::Dynamic> QS = (w * q_st).asDiagonal() * S_[k];
      P_.noalias() += S_[k].transpose() * QS;
      q_.noalias() += QS.transpose() * d;
    }
    // rate: (u_k - u_{k-1}) with u_{-1} = previous command
    for (int k = 0; k < N_; ++k)
      for (int j = 0; j < 2; ++j) {
        const int i = 2 * k + j;
        P_(i, i) += r_rate[j];
        if (k > 0) {
          P_(i - 2, i - 2) += r_rate[j];
          P_(i, i - 2) -= r_rate[j];
          P_(i - 2, i) -= r_rate[j];
        }
      }
    q_[0] -= r_rate[0] * v_prev;
    q_[1] -= r_rate[1] * d_prev;
    P_ *= 2.0;
    q_ *= 2.0;
    P_.diagonal().array() += 1e-9;
    // --- bounds
    for (int k = 0; k < N_; ++k) {
      l_[2 * k] = set_.v_min;
      u_[2 * k] = std::max(set_.v_min, std::min(set_.v_max, cap_at(k)));
      l_[2 * k + 1] = set_.delta_lo;
      u_[2 * k + 1] = set_.delta_hi;
      double rv = set_.v_rate_max * dt;
      const double rd = set_.delta_rate_max * dt;
      if (set_.grip_accel > 0.0) {
        // friction circle where this speed change acts (after the speed channel's extra delay)
        const Eigen::Vector4d& xa = xn_[std::min(k + nd, N_)];
        const double a_lat =
            xa[2] * xa[2] * std::fabs(std::tan(xa[3])) / (mod_.wheelbase * (1.0 + mod_.understeer * xa[2] * xa[2]));
        const double g2 = set_.grip_accel * set_.grip_accel;
        // never below 20 % of the grip: the car must still be able to slow down
        const double a_lon = set_.drive_axle_load * std::sqrt(std::max(g2 - a_lat * a_lat, 0.04 * g2));
        rv = std::min(rv, a_lon * dt);
      }
      l_[n_ + 2 * k] = (k == 0 ? v_prev : 0.0) - rv;
      u_[n_ + 2 * k] = (k == 0 ? v_prev : 0.0) + rv;
      l_[n_ + 2 * k + 1] = (k == 0 ? d_prev : 0.0) - rd;
      u_[n_ + 2 * k + 1] = (k == 0 ? d_prev : 0.0) + rd;
    }
    // the cap may fall faster than the rate limit allows: the cap wins (as the IRL node)
    if (l_[n_] > u_[0]) l_[n_] = u_[0] - 1e-9;
    const QpResult r = qp_.solve(P_, q_, A_, l_, u_);
    out.qp_iterations += r.iterations;
    out.qp_converged = r.converged;
    const Eigen::VectorXd& z = qp_.solution();
    if (!z.allFinite()) {
      out.ok = false;
      have_nominal_ = false;
      return out;
    }
    u_nom_ = z;
  }
  out.ok = true;
  out.v_cmd = u_nom_[0];
  out.delta_eff = u_nom_[1];
  rollout(x0, s0, path, v_prev);
  out.x_pred = xn_;
  out.s_pred = sn_;
  // receding horizon: the next call starts one stage later
  for (int k = 0; k + 1 < N_; ++k) u_nom_.segment<2>(2 * k) = u_nom_.segment<2>(2 * (k + 1));
  qp_.shift_warm_start(2);
  return out;
}

// ============================================================= actuator model
double LaggedActuator::value_at(double t) const {
  double x = x_, tt = t_, u = u_;
  for (const auto& p : pending_) {
    if (p.first > t) break;
    if (p.first > tt) {
      x = gain_ * u + (x - gain_ * u) * std::exp(-(p.first - tt) / tau_);
      tt = p.first;
    }
    u = p.second;
  }
  if (t > tt) x = gain_ * u + (x - gain_ * u) * std::exp(-(t - tt) / tau_);
  return x;
}

void LaggedActuator::advance(double t) {
  if (t <= t_) return;
  x_ = value_at(t);
  while (!pending_.empty() && pending_.front().first <= t) {
    u_ = pending_.front().second;
    pending_.pop_front();
  }
  t_ = t;
}

// ================================================================ controller
TrackingController::TrackingController(const ControllerParams& p)
    : p_(p), mpc_(p.model, p.mpc), grip_(p.grip_nominal), understeer_(p.model.understeer) {
  steer_.configure(p_.steer_dead_time, p_.model.steer_tau, 1.0);
  speed_.configure(p_.speed_dead_time, p_.model.speed_tau, p_.model.speed_gain);
}

void TrackingController::set_path(const std::vector<double>& x, const std::vector<double>& y) {
  path_ = std::make_unique<PathGeometry>(x, y);
  mpc_.reset();
}

double TrackingController::steer_to_eff(double d) const {
  if (!p_.use_steer_map) return d;
  const auto& c = p_.steer_map_cmd;
  const auto& e = p_.steer_map_eff;
  const size_t n = c.size();
  size_t i = 0;
  if (d <= c.front()) i = 0;
  else if (d >= c.back()) i = n - 2;
  else while (i + 2 < n && d > c[i + 1]) ++i;
  return e[i] + (d - c[i]) * (e[i + 1] - e[i]) / (c[i + 1] - c[i]);
}

double TrackingController::eff_to_steer(double de) const {
  if (!p_.use_steer_map) return de;
  const auto& c = p_.steer_map_cmd;
  const auto& e = p_.steer_map_eff;
  const size_t n = e.size();
  size_t i = 0;
  if (de <= e.front()) i = 0;
  else if (de >= e.back()) i = n - 2;
  else while (i + 2 < n && de > e[i + 1]) ++i;
  return c[i] + (de - e[i]) * (c[i + 1] - c[i]) / (e[i + 1] - e[i]);
}

void TrackingController::on_imu(double t, double r) {
  imu_.emplace_back(t, r);
  while (!imu_.empty() && imu_.front().first < t - 2.0) imu_.pop_front();
}

void TrackingController::on_speed(double t, double v) {
  vel_.emplace_back(t, v);
  while (!vel_.empty() && vel_.front().first < t - 2.0) vel_.pop_front();
}

double TrackingController::sample(const std::deque<std::pair<double, double>>& b, double t) const {
  if (b.empty()) return 0.0;
  if (t <= b.front().first) return b.front().second;
  if (t >= b.back().first) return b.back().second;
  auto it = std::lower_bound(b.begin(), b.end(), t, [](const std::pair<double, double>& p, double v) { return p.first < v; });
  const auto& p1 = *it;
  const auto& p0 = *(it - 1);
  const double w = (t - p0.first) / std::max(p1.first - p0.first, 1e-9);
  return p0.second + w * (p1.second - p0.second);
}

void TrackingController::on_command_sent(double t, double v_cmd, double d_cmd) {
  if (!steer_.initialised()) steer_.reset(t, 0.0, 0.0);
  if (!speed_.initialised()) speed_.reset(t, vel_.empty() ? 0.0 : vel_.back().second, 0.0);
  if (pending_arrival_ >= 0.0) {  // the command computed for the last pose
    const double dly = std::max(0.0, t - pending_arrival_);
    compute_ema_ = compute_ema_ > 0.0 ? 0.9 * compute_ema_ + 0.1 * dly : dly;
    pending_arrival_ = -1.0;
  }
  steer_.command(t, d_cmd);  // servo-command radians: lag first, then the map (as identified)
  speed_.command(t, v_cmd);
  last_v_cmd_ = v_cmd;
  last_d_cmd_ = d_cmd;
}

bool TrackingController::imu_mean(double t0, double t1, double& r) const {
  if (imu_.empty() || imu_.back().first < t1 || imu_.front().first > t0) return false;
  double r_sum = 0.0;
  int n = 0;
  for (const auto& s : imu_)
    if (s.first >= t0 && s.first <= t1) {
      r_sum += s.second;
      ++n;
    }
  if (n < 5) return false;
  r = r_sum / n;
  return true;
}

double TrackingController::yaw_rate_of(double v, double delta) const {
  return v * std::tan(delta) / (p_.model.wheelbase * (1.0 + understeer_ * v * v));
}

void TrackingController::update_grip(double t_arr) {
  if ((!p_.grip_adaptation && p_.understeer_tau <= 0.0) || vel_.empty()) return;
  const double dt = last_grip_t_ < 0.0 ? 0.0 : std::min(t_arr - last_grip_t_, 0.5);
  last_grip_t_ = t_arr;
  const double t1 = t_arr - 0.05, t0 = t1 - 0.1, tm = 0.5 * (t0 + t1);
  double r = 0.0;
  if (!imu_mean(t0, t1, r)) return;
  const double v = sample(vel_, tm);
  const double delta = steer_to_eff(steer_.value_at(tm)) + steer_offset_;
  const double r_model = yaw_rate_of(v, delta);
  const double tol = p_.slip_rel * std::fabs(r_model) + p_.slip_abs;
  // the car turns less than the model (front saturated: plows) while asked for a real turn, or
  // more (rear saturated: spins)
  const double sgn = r_model >= 0.0 ? 1.0 : -1.0;
  const bool plow = std::fabs(v * r_model) > 0.1 * 9.81 && sgn * (r_model - r) > tol;
  const bool spin = std::fabs(r) > 0.3 && sgn * (r - r_model) > tol;
  const bool slip = v > p_.slip_min_speed && (plow || spin);

  if (p_.grip_adaptation) {
    if (slip) {
      if (slip_count_ == 0) {
        grip_before_ = grip_;
        episode_max_ = 0.0;
      }
      // two windows in a row (0.2 s): one-off gyro / disturbance spikes are not a slide. The grip is
      // the most the car reached during the slide (transients through the slide reach less).
      episode_max_ = std::max(episode_max_, std::fabs(v * r) / 9.81);
      if (++slip_count_ >= 2) grip_ = std::clamp(std::min(grip_before_, episode_max_), p_.grip_min, p_.grip_nominal);
    } else {
      slip_count_ = 0;
      grip_ += (p_.grip_nominal - grip_) * (1.0 - std::exp(-dt / p_.grip_recover_tau));
    }
    const double g = p_.grip_margin * grip_ * 9.81;
    mpc_.set_grip_limits(std::min(p_.mpc.a_lat_max, g), std::min(p_.mpc.v_rate_max, g * p_.mpc.drive_axle_load), g);
  }
  if (p_.understeer_tau > 0.0 && !slip && v > 1.2 && std::fabs(r) > 0.3 && std::fabs(std::tan(delta)) > 0.05 &&
      r * delta > 0.0) {
    // r = v tan(d) / (L (1 + k v^2))  ->  k = (v tan(d) / (L r) - 1) / v^2
    const double k = (v * std::tan(delta) / (p_.model.wheelbase * r) - 1.0) / (v * v);
    understeer_ += (std::clamp(k, 0.0, p_.understeer_max) - understeer_) * (1.0 - std::exp(-dt / p_.understeer_tau));
    mpc_.set_understeer(understeer_);
  }
}

void TrackingController::update_steer_offset(double t_arr) {
  if (p_.steer_offset_tau <= 0.0 || imu_.empty() || vel_.empty()) return;
  // the last 0.1 s, 50 ms back so that the IMU and VESC samples around it have arrived
  const double t1 = t_arr - 0.05, t0 = t1 - 0.1;
  double r_mean = 0.0;
  if (!imu_mean(t0, t1, r_mean)) return;
  // a sliding car says nothing about the servo trim
  if (slip_count_ > 0) return;
  const double tm = 0.5 * (t0 + t1), v = sample(vel_, tm);
  const double d_model = steer_to_eff(steer_.value_at(tm));
  const double dt = last_offset_t_ < 0.0 ? 0.0 : std::min(t_arr - last_offset_t_, 0.5);
  last_offset_t_ = t_arr;
  if (v < p_.steer_offset_min_speed || std::fabs(d_model) > p_.steer_offset_max_delta) return;
  // the angle that explains the measured yaw rate, minus the one the model assumes
  const double implied = std::atan(r_mean * p_.model.wheelbase / v) - d_model;
  steer_offset_ += (implied - steer_offset_) * (1.0 - std::exp(-dt / p_.steer_offset_tau));
  steer_offset_ = std::clamp(steer_offset_, -p_.steer_offset_max, p_.steer_offset_max);
}

ControlOutput TrackingController::on_pose(double t_arr, double px, double py, double yaw, double v_ref,
                                          double pose_age) {
  ControlOutput o;
  if (!path_) return o;
  if (vel_.empty()) {
    o.status = Status::kNoSpeed;
    return o;
  }
  PathGeometry& path = *path_;
  if (!steer_.initialised()) steer_.reset(t_arr - 1.0, 0.0, 0.0);
  if (!speed_.initialised()) speed_.reset(t_arr - 1.0, vel_.back().second, 0.0);
  steer_.advance(t_arr - 1.0);
  speed_.advance(t_arr - 1.0);

  o.meas = path.project(px, py, yaw);
  if (std::fabs(o.meas.e_lat) > p_.abort_e_lat || std::fabs(o.meas.e_psi) > p_.abort_e_psi) {
    o.status = Status::kAbort;
    mpc_.reset();
    return o;
  }
  // ---- latency compensation
  const double a = p_.model.lever_arm;
  const double t_pos = t_arr - pose_age - p_.pose_latency_position;
  const double t_hdg = t_arr - pose_age - p_.pose_latency_heading;
  const double t_start = t_arr + std::max(p_.compute_delay, compute_ema_) + p_.steer_dead_time;
  const double h = 0.005;
  const bool imu_ok = p_.use_imu && !imu_.empty() && imu_.back().first > t_arr - 0.2;
  update_grip(t_arr);
  update_steer_offset(t_arr);
  auto delta_at = [&](double t) { return steer_to_eff(steer_.value_at(t)) + steer_offset_; };
  auto yaw_rate_model = [&](double t, double v) { return yaw_rate_of(v, delta_at(t)); };
  // heading at the position's time: back-integrate the IMU over [t_pos, t_hdg]
  double psi = yaw + p_.heading_bias;
  for (double t = t_hdg; t > t_pos + 1e-9; t -= h) {
    const double hh = std::min(h, t - t_pos);
    psi -= hh * (imu_ok ? sample(imu_, t - 0.5 * hh) : yaw_rate_model(t - 0.5 * hh, sample(vel_, t - 0.5 * hh)));
  }
  double rx = px - a * std::cos(psi), ry = py - a * std::sin(psi);  // rear axle
  // [t_pos, t_arr]: measured speed and yaw rate
  for (double t = t_pos; t < t_arr - 1e-9; t += h) {
    const double hh = std::min(h, t_arr - t), tm = t + 0.5 * hh;
    const double v = sample(vel_, tm);
    const double r = imu_ok ? sample(imu_, tm) : yaw_rate_model(tm, v);
    const double pm = psi + 0.5 * hh * r;
    rx += hh * v * std::cos(pm);
    ry += hh * v * std::sin(pm);
    psi += hh * r;
  }
  // [t_arr, t_start]: the commands already sent, through the actuator model
  const double v_arr = sample(vel_, t_arr);
  const double vm_arr = speed_.value_at(t_arr);
  for (double t = t_arr; t < t_start - 1e-9; t += h) {
    const double hh = std::min(h, t_start - t), tm = t + 0.5 * hh;
    const double v = v_arr + speed_.value_at(tm) - vm_arr;
    const double r = yaw_rate_of(v, delta_at(tm));
    const double pm = psi + 0.5 * hh * r;
    rx += hh * v * std::cos(pm);
    ry += hh * v * std::sin(pm);
    psi += hh * r;
  }
  const Projection pr = path.project(rx + a * std::cos(psi), ry + a * std::sin(psi), psi);
  o.x0 << pr.e_lat, pr.e_psi, v_arr + speed_.value_at(t_start) - vm_arr, delta_at(t_start);
  o.s0 = pr.s;
  o.t_start = t_start;
  // back to the measured point for the next local projection
  path.project(px, py, yaw);

  // ---- MPC
  o.mpc = mpc_.solve(o.x0, o.s0, v_ref, path, last_v_cmd_, steer_to_eff(last_d_cmd_) + steer_offset_);
  if (!o.mpc.ok) {
    o.status = Status::kAbort;
    return o;
  }
  // command limits exactly as the IRL node: box, rate per control period
  const double rv = mpc_.settings().v_rate_max * p_.control_period, rd = p_.mpc.delta_rate_max * p_.control_period;
  o.v_cmd = std::min(std::max(o.mpc.v_cmd, std::max(p_.mpc.v_min, last_v_cmd_ - rv)), std::min(p_.mpc.v_max, last_v_cmd_ + rv));
  const double d = eff_to_steer(o.mpc.delta_eff - steer_offset_);
  o.delta_cmd = std::min(std::max(d, std::max(-p_.delta_cmd_max, last_d_cmd_ - rd)), std::min(p_.delta_cmd_max, last_d_cmd_ + rd));
  o.status = Status::kOk;
  pending_arrival_ = t_arr;
  return o;
}

}  // namespace mpc_tc
