#include "rc_localization/ekf_core.hpp"

#include <algorithm>
#include <cmath>

namespace rc_localization
{

double normalize_angle(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

// ---------------------------------------------------------------- Ekf

void Ekf::predict(double dt)
{
  // Split long gaps so the linearisation stays accurate in sharp turns.
  while (dt > 1e-9) {
    const double h = std::min(dt, p_.max_predict_dt);
    predict_step(h);
    dt -= h;
  }
}

double Ekf::lateral_velocity() const
{
  return p_.lateral_dynamics ? x(VY) : p_.lever_arm * x(WZ);
}

void Ekf::predict_step(double dt)
{
  const double th = x(YAW) + x(YAW_BIAS);
  const double c = std::cos(th), s = std::sin(th);
  const double v = x(VX), w = x(WZ), a = p_.lever_arm;
  const double vl = lateral_velocity();

  // Jacobian F = df/dx, evaluated at the state before the step.
  StateMat F = StateMat::Identity();
  const double dx_dth = (-v * s - vl * c) * dt;
  const double dy_dth = (v * c - vl * s) * dt;
  F(X, YAW) = dx_dth;   F(X, YAW_BIAS) = dx_dth;
  F(Y, YAW) = dy_dth;   F(Y, YAW_BIAS) = dy_dth;
  F(X, VX) = c * dt;
  F(Y, VX) = s * dt;
  F(YAW, WZ) = dt;
  // d(position)/d(v_lat) * d(v_lat)/d(state)
  const int lat = p_.lateral_dynamics ? VY : WZ;
  const double dvl = p_.lateral_dynamics ? 1.0 : a;
  F(X, lat) += -s * dt * dvl;
  F(Y, lat) += c * dt * dvl;

  x(X) += (v * c - vl * s) * dt;
  x(Y) += (v * s + vl * c) * dt;
  x(YAW) = normalize_angle(x(YAW) + w * dt);

  StateVec q;
  q << p_.q_pos, p_.q_pos, p_.q_yaw, p_.q_yaw_bias, p_.q_vx, p_.q_wz, p_.q_gyro_bias, p_.q_speed_scale, 0.0, 0.0, 0.0,
    p_.q_vo_scale;

  if (p_.lateral_dynamics) {
    const double vy = x(VY), ay = x(AY), k = 1.0 / p_.slip_tau;
    // vy' = vy + (ay - vx wz - (vy - a wz) / tau) dt
    F(VY, VY) = 1.0 - k * dt;
    F(VY, AY) = dt;
    F(VY, VX) = -w * dt;
    F(VY, WZ) = (-v + a * k) * dt;
    x(VY) = vy + (ay - v * w - k * (vy - a * w)) * dt;
    q(VY) = p_.q_vy;
    q(AY) = p_.q_ay;
    q(AY_BIAS) = p_.q_ay_bias;
  }
  const StateMat Q = (q.array().square() * dt).matrix().asDiagonal();

  P = F * P * F.transpose() + Q;
}

bool Ekf::update_pose(const PoseMeas & m, double * d2)
{
  Eigen::Matrix<double, 3, NX> H = Eigen::Matrix<double, 3, NX>::Zero();
  H(0, X) = 1.0;
  H(1, Y) = 1.0;
  H(2, YAW) = 1.0;

  Eigen::Vector3d h(x(X), x(Y), x(YAW));
  if (m.position_delay != 0.0) {
    // h_pos = p - d u(x), u = world velocity of base_link
    const double d = m.position_delay;
    const double th = x(YAW) + x(YAW_BIAS);
    const double c = std::cos(th), s = std::sin(th), v = x(VX), vl = lateral_velocity();
    h(0) -= d * (v * c - vl * s);
    h(1) -= d * (v * s + vl * c);
    const double dux_dth = -v * s - vl * c, duy_dth = v * c - vl * s;
    H(0, YAW) = H(0, YAW_BIAS) = -d * dux_dth;
    H(1, YAW) = H(1, YAW_BIAS) = -d * duy_dth;
    H(0, VX) = -d * c;
    H(1, VX) = -d * s;
    const int lat = p_.lateral_dynamics ? VY : WZ;
    const double dvl = p_.lateral_dynamics ? 1.0 : p_.lever_arm;
    H(0, lat) = d * s * dvl;
    H(1, lat) = -d * c * dvl;
  }
  Eigen::Vector3d y = m.z - h;
  y(2) = normalize_angle(y(2));

  Eigen::Matrix3d R = m.R;
  if (m.time_stddev > 0.0) {
    const double th = x(YAW) + x(YAW_BIAS);
    const double vl = lateral_velocity();
    const Eigen::Vector3d u(x(VX) * std::cos(th) - vl * std::sin(th),
                            x(VX) * std::sin(th) + vl * std::cos(th), x(WZ));
    R += m.time_stddev * m.time_stddev * u * u.transpose();
  }

  const Eigen::Matrix3d S = H * P * H.transpose() + R;
  const Eigen::Matrix3d S_inv = S.inverse();
  const double maha = y.dot(S_inv * y);
  if (d2) *d2 = maha;
  if (p_.pose_gate > 0.0 && maha > p_.pose_gate) return false;

  const Eigen::Matrix<double, NX, 3> K = P * H.transpose() * S_inv;
  x += K * y;
  // Joseph form: keeps P symmetric positive definite despite rounding.
  const StateMat I_KH = StateMat::Identity() - K * H;
  P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
  finish_update();
  return true;
}

bool Ekf::update_twist(const TwistMeas & m, double * d2)
{
  Eigen::Matrix<double, 2, NX> H = Eigen::Matrix<double, 2, NX>::Zero();
  H(0, VX) = x(SPEED_SCALE);
  H(0, SPEED_SCALE) = x(VX);
  H(1, WZ) = 1.0;
  H(1, GYRO_BIAS) = 1.0;

  const Eigen::Vector2d h(x(SPEED_SCALE) * x(VX), x(WZ) + x(GYRO_BIAS));
  const Eigen::Vector2d y = m.z - h;

  const Eigen::Matrix2d S = H * P * H.transpose() + m.R;
  const Eigen::Matrix2d S_inv = S.inverse();
  const double maha = y.dot(S_inv * y);
  if (d2) *d2 = maha;
  if (p_.twist_gate > 0.0 && maha > p_.twist_gate) return false;

  const Eigen::Matrix<double, NX, 2> K = P * H.transpose() * S_inv;
  x += K * y;
  const StateMat I_KH = StateMat::Identity() - K * H;
  P = I_KH * P * I_KH.transpose() + K * m.R * K.transpose();
  finish_update();
  return true;
}

bool Ekf::update_accel(const AccelMeas & m, double * d2)
{
  if (!p_.lateral_dynamics) return false;
  Eigen::Matrix<double, 1, NX> H = Eigen::Matrix<double, 1, NX>::Zero();
  H(0, AY) = 1.0;
  H(0, AY_BIAS) = 1.0;

  const double y = m.z - (x(AY) + x(AY_BIAS));
  const double S = (H * P * H.transpose())(0, 0) + m.R;
  if (d2) *d2 = y * y / S;

  const Eigen::Matrix<double, NX, 1> K = P * H.transpose() / S;
  x += K * y;
  const StateMat I_KH = StateMat::Identity() - K * H;
  P = I_KH * P * I_KH.transpose() + K * m.R * K.transpose();
  finish_update();
  return true;
}

bool Ekf::update_vo(const VoMeas & m, double * d2)
{
  const double s = x(VO_SCALE), v = x(VX), w = x(WZ), l = p_.vo_lever;
  const double vl = lateral_velocity();
  Eigen::Matrix<double, 2, NX> H = Eigen::Matrix<double, 2, NX>::Zero();
  H(0, VX) = s;
  H(0, VO_SCALE) = v;
  H(1, VO_SCALE) = vl + l * w;
  if (p_.lateral_dynamics) {
    H(1, VY) = s;
    H(1, WZ) = s * l;
  } else {
    H(1, WZ) = s * (p_.lever_arm + l);
  }
  const Eigen::Vector2d h(s * v, s * (vl + l * w));
  const Eigen::Vector2d y = m.z - h;

  const Eigen::Matrix2d S = H * P * H.transpose() + m.R;
  const Eigen::Matrix2d S_inv = S.inverse();
  const double maha = y.dot(S_inv * y);
  if (d2) *d2 = maha;
  if (p_.vo_gate > 0.0 && maha > p_.vo_gate) return false;

  const Eigen::Matrix<double, NX, 2> K = P * H.transpose() * S_inv;
  x += K * y;
  const StateMat I_KH = StateMat::Identity() - K * H;
  P = I_KH * P * I_KH.transpose() + K * m.R * K.transpose();
  finish_update();
  return true;
}

void Ekf::finish_update()
{
  x(YAW) = normalize_angle(x(YAW));
  x(YAW_BIAS) = std::clamp(x(YAW_BIAS), -p_.max_yaw_bias, p_.max_yaw_bias);
  x(GYRO_BIAS) = std::clamp(x(GYRO_BIAS), -p_.max_gyro_bias, p_.max_gyro_bias);
  x(SPEED_SCALE) = std::clamp(x(SPEED_SCALE), p_.min_speed_scale, p_.max_speed_scale);
  x(AY_BIAS) = std::clamp(x(AY_BIAS), -p_.max_ay_bias, p_.max_ay_bias);
  x(VO_SCALE) = std::clamp(x(VO_SCALE), p_.min_vo_scale, p_.max_vo_scale);
  P = 0.5 * (P + P.transpose());
}

// ---------------------------------------------------------------- DelayedEkf

void DelayedEkf::initialize(double t, const StateVec & x0, const StateMat & P0)
{
  hist_.clear();
  Entry e;
  e.m.t = t;
  e.m.type = Measurement::Type::INIT;
  e.x = x0;
  e.P = P0;
  hist_.push_back(e);
}

void DelayedEkf::process(Entry & e, double t_prev)
{
  ekf_.predict(e.m.t - t_prev);
  e.d2 = 0.0;
  switch (e.m.type) {
    case Measurement::Type::POSE:
      e.accepted = ekf_.update_pose(e.m.pose, &e.d2);
      break;
    case Measurement::Type::TWIST:
      e.accepted = ekf_.update_twist(e.m.twist, &e.d2);
      break;
    case Measurement::Type::ACCEL:
      e.accepted = ekf_.update_accel(e.m.accel, &e.d2);
      break;
    case Measurement::Type::VO:
      e.accepted = ekf_.update_vo(e.m.vo, &e.d2);
      break;
    case Measurement::Type::INIT:
      break;
  }
  e.x = ekf_.x;
  e.P = ekf_.P;
}

DelayedEkf::Result DelayedEkf::add(const Measurement & m, double * d2, int * replayed)
{
  if (replayed) *replayed = 0;
  if (hist_.empty()) return Result::NOT_INITIALIZED;
  if (m.t < hist_.front().m.t) return Result::TOO_OLD;

  // Insert after every entry with t <= m.t (search from the back: most
  // measurements are the newest, delayed poses land a few entries back).
  std::size_t k = hist_.size();
  while (k > 1 && hist_[k - 1].m.t > m.t) --k;

  Entry e;
  e.m = m;
  hist_.insert(hist_.begin() + static_cast<std::ptrdiff_t>(k), e);

  // Restore the state just before m, then (re)process m and everything after.
  ekf_.x = hist_[k - 1].x;
  ekf_.P = hist_[k - 1].P;
  for (std::size_t i = k; i < hist_.size(); ++i) {
    process(hist_[i], hist_[i - 1].m.t);
  }
  if (replayed) *replayed = static_cast<int>(hist_.size() - k - 1);

  const Entry & added = hist_[k];
  if (d2) *d2 = added.d2;
  const Result r = added.accepted ? Result::APPLIED : Result::REJECTED;
  prune();
  return r;
}

void DelayedEkf::prune()
{
  // Keep at least one entry as the restore point for late measurements.
  const double t_min = hist_.back().m.t - history_;
  while (hist_.size() > 1 && hist_[1].m.t < t_min) hist_.pop_front();
}

void DelayedEkf::predict_to(double t, double max_extrapolation, StateVec & x, StateMat & P) const
{
  Ekf e(p_);
  e.x = hist_.back().x;
  e.P = hist_.back().P;
  const double dt = std::min(t - hist_.back().m.t, max_extrapolation);
  if (dt > 0.0) e.predict(dt);
  x = e.x;
  P = e.P;
}

void DelayedEkf::state_at(double t, StateVec & x, StateMat & P) const
{
  auto it = hist_.rbegin();
  while (it != hist_.rend() && it->m.t > t) ++it;
  if (it == hist_.rend()) it = std::prev(hist_.rend());
  Ekf e(p_);
  e.x = it->x;
  e.P = it->P;
  if (t > it->m.t) e.predict(t - it->m.t);
  x = e.x;
  P = e.P;
}

}  // namespace rc_localization
