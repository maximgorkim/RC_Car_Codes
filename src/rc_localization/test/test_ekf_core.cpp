// Closed-loop tests of the EKF core against a simulated car with known truth.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "rc_localization/ekf_core.hpp"

using namespace rc_localization;

namespace
{

struct Truth
{
  double t, x, y, yaw, v, w;
};

// Rear-axle kinematics with base_link lever_arm ahead, speed up to v_max with
// alternating left/right turns: the same model the filter assumes.
std::vector<Truth> simulate(double duration, double dt, double v_max, double lever)
{
  std::vector<Truth> out;
  Truth s{0, 0, 0, 0, 0, 0};
  for (double t = 0; t <= duration; t += dt) {
    s.t = t;
    s.v = v_max * std::min(1.0, t / 4.0) * (0.75 + 0.25 * std::sin(0.4 * t));
    // Lateral acceleration v*w kept <= ~8 m/s^2.
    s.w = (s.v > 0.1 ? std::min(1.2, 8.0 / s.v) : 0.0) * std::sin(0.7 * t);
    out.push_back(s);
    const double c = std::cos(s.yaw), sn = std::sin(s.yaw);
    s.x += (s.v * c - lever * s.w * sn) * dt;
    s.y += (s.v * sn + lever * s.w * c) * dt;
    s.yaw = normalize_angle(s.yaw + s.w * dt);
  }
  return out;
}

struct Result
{
  double rms_pos = 0, max_pos = 0, rms_yaw = 0;
  int rejected = 0, outliers_rejected = 0;
  StateVec x_final;
};

// Runs the filter like the ROS node does: twists at 200 Hz in time order, poses
// at 8 Hz delivered `delay` seconds late. Errors are measured on the newest
// state (what the node publishes) against the truth at that time.
Result run(double delay, bool compensate, bool gate = true, double v_max = 10.0, unsigned seed = 1)
{
  const double dt = 0.005, lever = 0.17;
  const double gyro_bias = 0.012, scale = 1.1;
  const auto truth = simulate(40.0, dt, v_max, lever);

  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, 1.0);

  EkfParams p;
  p.lever_arm = lever;
  if (!gate) p.pose_gate = 0.0;
  DelayedEkf ekf(p, 1.0);
  StateVec x0 = StateVec::Zero();
  x0(SPEED_SCALE) = 1.0;
  StateMat P0 = StateMat::Zero();
  P0.diagonal() << 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.02 * 0.02, 0.15 * 0.15, 0.0, 0.0, 0.0;
  ekf.initialize(0.0, x0, P0);

  struct Pending { double arrive; Measurement m; bool outlier; };
  std::vector<Pending> queue;

  Result r;
  int n_err = 0;
  double vesc_speed = 0.0;
  for (std::size_t i = 1; i < truth.size(); ++i) {
    const Truth & s = truth[i];

    if (i % 4 == 0) vesc_speed = scale * s.v + 0.05 * n(rng);   // VESC, 50 Hz

    Measurement tw;
    tw.t = s.t;
    tw.type = Measurement::Type::TWIST;
    tw.twist.z << vesc_speed, s.w + gyro_bias + 0.01 * n(rng);
    tw.twist.R = Eigen::Vector2d(0.1 * 0.1, 0.02 * 0.02).asDiagonal();
    ekf.add(tw);

    if (i % 25 == 0) {                                             // LiDAR pose, 8 Hz
      Pending pd;
      pd.m.t = compensate ? s.t : s.t + delay;  // without compensation: stamped on arrival
      pd.m.type = Measurement::Type::POSE;
      pd.m.pose.z << s.x + 0.03 * n(rng), s.y + 0.03 * n(rng), s.yaw + 0.01 * n(rng);
      pd.outlier = (i % 2000 == 0);
      if (pd.outlier) pd.m.pose.z(0) += 2.0;
      pd.m.pose.R = Eigen::Vector3d(0.03 * 0.03, 0.03 * 0.03, 0.01 * 0.01).asDiagonal();
      pd.arrive = s.t + delay;
      queue.push_back(pd);
    }
    for (auto it = queue.begin(); it != queue.end();) {
      if (it->arrive <= s.t + 1e-9) {
        if (ekf.add(it->m) == DelayedEkf::Result::REJECTED) {
          ++r.rejected;
          if (it->outlier) ++r.outliers_rejected;
        }
        it = queue.erase(it);
      } else {
        ++it;
      }
    }

    if (s.t > 8.0) {
      const StateVec & x = ekf.state();
      const double e = std::hypot(x(X) - s.x, x(Y) - s.y);
      const double ey = normalize_angle(x(YAW) - s.yaw);
      r.rms_pos += e * e;
      r.rms_yaw += ey * ey;
      r.max_pos = std::max(r.max_pos, e);
      ++n_err;
    }
  }
  r.rms_pos = std::sqrt(r.rms_pos / n_err);
  r.rms_yaw = std::sqrt(r.rms_yaw / n_err);
  r.x_final = ekf.state();
  return r;
}

}  // namespace

TEST(EkfCore, NormalizeAngle)
{
  EXPECT_NEAR(normalize_angle(3 * M_PI), M_PI, 1e-9);
  EXPECT_NEAR(normalize_angle(-0.5), -0.5, 1e-12);
}

TEST(EkfCore, PredictStraightLine)
{
  EkfParams p;
  Ekf e(p);
  e.x.setZero();
  e.x(VX) = 10.0;
  e.x(SPEED_SCALE) = 1.0;
  e.predict(0.5);
  EXPECT_NEAR(e.x(X), 5.0, 1e-9);
  EXPECT_NEAR(e.x(Y), 0.0, 1e-9);
}

void check_jacobian(bool lateral)
{
  EkfParams p;
  p.lateral_dynamics = lateral;
  p.q_pos = p.q_yaw = p.q_yaw_bias = p.q_vx = p.q_wz = p.q_gyro_bias = p.q_speed_scale = 0.0;
  p.q_vy = p.q_ay = p.q_ay_bias = 0.0;
  StateVec x0;
  x0 << 1.0, -2.0, 0.7, 0.05, 6.0, 0.8, 0.01, 1.05, 0.4, 3.0, 0.2;
  const double dt = 0.01, eps = 1e-6;

  // With Q = 0 and P = I, one step gives P' = F F^T with the analytic F.
  Ekf base(p);
  base.x = x0;
  base.P = StateMat::Identity();
  base.predict(dt);

  // Numeric Jacobian of the state transition.
  StateMat F_num;
  for (int j = 0; j < NX; ++j) {
    Ekf a(p), b(p);
    a.x = x0; b.x = x0;
    a.x(j) += eps; b.x(j) -= eps;
    a.predict(dt); b.predict(dt);
    F_num.col(j) = (a.x - b.x) / (2 * eps);
  }
  EXPECT_LT((F_num * F_num.transpose() - base.P).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(EkfCore, JacobianMatchesNumeric) { check_jacobian(false); }
TEST(EkfCore, JacobianMatchesNumericLateral) { check_jacobian(true); }

// Pose with an older position: h = [p - d u(x), yaw]. The measurement Jacobian
// is recovered from the update itself (P = s I, s tiny, R = I: dx = s H^T y)
// and compared with the numeric derivative of h.
void check_position_delay(bool lateral)
{
  EkfParams p;
  p.lateral_dynamics = lateral;
  p.pose_gate = 0.0;
  StateVec x0;
  x0 << 1.0, -2.0, 0.7, 0.05, 6.0, 0.8, 0.01, 1.05, 0.4, 3.0, 0.2;
  const double d = 0.04;
  auto h = [&](const StateVec & x) {
    const double th = x(YAW) + x(YAW_BIAS);
    const double vl = lateral ? x(VY) : p.lever_arm * x(WZ);
    return Eigen::Vector3d(x(X) - d * (x(VX) * std::cos(th) - vl * std::sin(th)),
                           x(Y) - d * (x(VX) * std::sin(th) + vl * std::cos(th)), x(YAW));
  };
  PoseMeas m;
  m.position_delay = d;
  m.R = Eigen::Matrix3d::Identity();

  // A measurement equal to h(x) changes nothing.
  Ekf e(p);
  e.x = x0;
  e.P = 1e-8 * StateMat::Identity();
  m.z = h(x0);
  ASSERT_TRUE(e.update_pose(m));
  EXPECT_LT((e.x - x0).cwiseAbs().maxCoeff(), 1e-12);

  for (int i = 0; i < 3; ++i) {
    Ekf f(p);
    f.x = x0;
    f.P = 1e-8 * StateMat::Identity();
    m.z = h(x0);
    m.z(i) += 1.0;
    ASSERT_TRUE(f.update_pose(m));
    const StateVec H_row = (f.x - x0) / 1e-8;
    StateVec H_num;
    for (int j = 0; j < NX; ++j) {
      StateVec a = x0, b = x0;
      a(j) += 1e-6;
      b(j) -= 1e-6;
      H_num(j) = (h(a)(i) - h(b)(i)) / 2e-6;
    }
    EXPECT_LT((H_row - H_num).cwiseAbs().maxCoeff(), 1e-3) << "row " << i;
  }
}

TEST(EkfCore, PositionDelayJacobian) { check_position_delay(false); }
TEST(EkfCore, PositionDelayJacobianLateral) { check_position_delay(true); }

TEST(EkfCore, OutOfOrderEqualsInOrder)
{
  // Adding a pose late (with replay) must give exactly the same result as
  // adding it on time.
  EkfParams p;
  StateVec x0 = StateVec::Zero();
  x0(SPEED_SCALE) = 1.0;
  StateMat P0 = StateMat::Identity() * 0.1;

  DelayedEkf a(p, 1.0), b(p, 1.0);
  a.initialize(0.0, x0, P0);
  b.initialize(0.0, x0, P0);

  Measurement pose;
  pose.t = 0.3;
  pose.type = Measurement::Type::POSE;
  pose.pose.z << 0.5, 0.1, 0.05;
  pose.pose.R = Eigen::Matrix3d::Identity() * 0.01;

  for (int i = 1; i <= 100; ++i) {
    Measurement tw;
    tw.t = i * 0.005;
    tw.type = Measurement::Type::TWIST;
    tw.twist.z << 2.0, 0.3;
    tw.twist.R = Eigen::Matrix2d::Identity() * 0.01;
    a.add(tw);
    b.add(tw);
    if (i == 60) a.add(pose);  // on time (t = 0.3)
  }
  int replayed = 0;
  b.add(pose, nullptr, &replayed);  // 0.2 s late
  EXPECT_EQ(replayed, 40);
  EXPECT_LT((a.state() - b.state()).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(EkfCore, TooOldMeasurementIsDropped)
{
  EkfParams p;
  DelayedEkf f(p, 0.5);
  StateVec x0 = StateVec::Zero();
  x0(SPEED_SCALE) = 1.0;
  f.initialize(0.0, x0, StateMat::Identity());
  for (int i = 1; i <= 400; ++i) {
    Measurement tw;
    tw.t = i * 0.005;
    tw.type = Measurement::Type::TWIST;
    tw.twist.R = Eigen::Matrix2d::Identity();
    f.add(tw);
  }
  Measurement pose;
  pose.t = 0.5;  // 1.5 s old, history is 0.5 s
  pose.type = Measurement::Type::POSE;
  EXPECT_EQ(f.add(pose), DelayedEkf::Result::TOO_OLD);
}

TEST(EkfClosedLoop, TenMetersPerSecondWithDelayedPose)
{
  const Result r = run(0.2, true);
  std::printf("delay-compensated: rms pos %.3f m, max %.3f m, rms yaw %.4f rad, rejected %d\n",
              r.rms_pos, r.max_pos, r.rms_yaw, r.rejected);
  EXPECT_LT(r.rms_pos, 0.05);
  EXPECT_LT(r.max_pos, 0.20);
  EXPECT_LT(r.rms_yaw, 0.02);
  // Biases are learned.
  EXPECT_NEAR(r.x_final(GYRO_BIAS), 0.012, 0.004);
  EXPECT_NEAR(r.x_final(SPEED_SCALE), 1.1, 0.03);
  // The 2 m jumps are rejected by the Mahalanobis gate, good poses are not.
  EXPECT_EQ(r.outliers_rejected, 3);
  EXPECT_LE(r.rejected, 6);
}

TEST(EkfClosedLoop, DelayCompensationBeatsStampingOnArrival)
{
  // Gate off for both, otherwise the late poses would simply be rejected.
  const Result good = run(0.2, true, false);
  const Result bad = run(0.2, false, false);
  std::printf("rms pos: compensated %.3f m, stamped-on-arrival %.3f m\n", good.rms_pos, bad.rms_pos);
  EXPECT_LT(good.rms_pos * 3.0, bad.rms_pos);
}

namespace
{

struct SlipResult
{
  double rms_all = 0, rms_slip = 0, max_slip = 0;
  int rejected = 0, n_slides = 0;
};

// Car at 6-8 m/s that slides sideways in its curves (lateral velocity bursts of
// up to 1.2 m/s, as in Gazebo at a_lat 6 m/s^2). The IMU measures the true
// lateral acceleration of base_link (body frame), with bias and noise.
SlipResult run_slip(bool lateral, double tau = -1.0, unsigned seed = 3)
{
  const double dt = 0.005, a = 0.17, delay = 0.2;
  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, 1.0);

  EkfParams p;
  p.lever_arm = a;
  p.lateral_dynamics = lateral;
  if (tau > 0) p.slip_tau = tau;
  DelayedEkf ekf(p, 1.0);
  StateVec x0 = StateVec::Zero();
  x0(SPEED_SCALE) = 1.0;
  StateMat P0 = StateMat::Zero();
  P0.diagonal() << 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 1e-4, 1e-4, 0.04, 4.0, 0.25;
  ekf.initialize(0.0, x0, P0);

  struct Pending { double arrive; Measurement m; };
  std::vector<Pending> queue;
  SlipResult r;
  int n_all = 0, n_slip = 0;
  double x = 0, y = 0, yaw = 0, vlat_prev = 0;
  for (int i = 1; i * dt <= 40.0; ++i) {
    const double t = i * dt;
    const double v = std::min(1.0, t / 3.0) * (7.0 + 1.0 * std::sin(0.3 * t));
    const double w = 1.2 * std::sin(0.8 * t);
    // a slide in the middle of every curve (|w| peaks every ~3.9 s)
    const double phase = std::fmod(t, M_PI / 0.8);
    const double slide = (t > 5.0 ? 1.2 : 0.0) * std::exp(-std::pow((phase - 0.5 * M_PI / 0.8) / 0.35, 2)) *
                         (std::sin(0.8 * t) > 0 ? -1.0 : 1.0);
    const double vlat = a * w + slide;
    const double ay_true = (vlat - vlat_prev) / dt + v * w;
    vlat_prev = vlat;
    const double c = std::cos(yaw), s = std::sin(yaw);
    x += (v * c - vlat * s) * dt;
    y += (v * s + vlat * c) * dt;
    yaw = normalize_angle(yaw + w * dt);

    Measurement tw;
    tw.t = t;
    tw.type = Measurement::Type::TWIST;
    tw.twist.z << v + 0.05 * n(rng), w + 0.01 * n(rng);   // the wheels still report v
    tw.twist.R = Eigen::Vector2d(0.1 * 0.1, 0.02 * 0.02).asDiagonal();
    ekf.add(tw);
    if (lateral && i > 1) {
      Measurement ac;
      ac.t = t;
      ac.type = Measurement::Type::ACCEL;
      ac.accel.z = ay_true + 0.3 + 0.5 * n(rng);
      ac.accel.R = 0.5 * 0.5;
      ekf.add(ac);
    }
    if (i % 25 == 0) {
      Pending pd;
      pd.m.t = t;
      pd.m.type = Measurement::Type::POSE;
      pd.m.pose.z << x + 0.03 * n(rng), y + 0.03 * n(rng), yaw + 0.01 * n(rng);
      pd.m.pose.R = Eigen::Vector3d(0.03 * 0.03, 0.03 * 0.03, 0.01 * 0.01).asDiagonal();
      pd.arrive = t + delay;
      queue.push_back(pd);
    }
    for (auto it = queue.begin(); it != queue.end();) {
      if (it->arrive <= t + 1e-9) {
        if (ekf.add(it->m) == DelayedEkf::Result::REJECTED) ++r.rejected;
        it = queue.erase(it);
      } else {
        ++it;
      }
    }
    if (t > 8.0) {
      const StateVec & xs = ekf.state();
      const double e = std::hypot(xs(X) - x, xs(Y) - y);
      r.rms_all += e * e;
      ++n_all;
      if (std::abs(slide) > 0.3) {
        r.rms_slip += e * e;
        r.max_slip = std::max(r.max_slip, e);
        ++n_slip;
      }
    }
  }
  r.rms_all = std::sqrt(r.rms_all / n_all);
  r.rms_slip = std::sqrt(r.rms_slip / n_slip);
  return r;
}

}  // namespace

TEST(EkfClosedLoop, SlipTauSweep)
{
  for (double tau : {0.3, 1.0, 2.0, 5.0, 20.0}) {
    const SlipResult r = run_slip(true, tau);
    std::printf("slip_tau %5.1f: rms %.3f slides %.3f max %.3f rejected %d\n", tau, r.rms_all, r.rms_slip, r.max_slip, r.rejected);
  }
}

TEST(EkfClosedLoop, LateralDynamicsTracksSlides)
{
  const SlipResult kin = run_slip(false);
  const SlipResult lat = run_slip(true);
  std::printf("sliding car: kinematic rms %.3f m (slides %.3f, max %.3f, rejected %d) | "
              "lateral dynamics rms %.3f m (slides %.3f, max %.3f, rejected %d)\n",
              kin.rms_all, kin.rms_slip, kin.max_slip, kin.rejected, lat.rms_all, lat.rms_slip, lat.max_slip,
              lat.rejected);
  EXPECT_LT(lat.rms_slip * 2.0, kin.rms_slip);
  EXPECT_LT(lat.max_slip, 0.20);   // kinematic: 0.76 m
  EXPECT_LE(lat.rejected, kin.rejected);
}

TEST(EkfClosedLoop, LateralDynamicsHarmlessWithoutSlip)
{
  // With the lateral model on but no slides, the result must match the
  // kinematic model (no regression on normal driving).
  const Result base = run(0.2, true);
  std::printf("no slip, kinematic: rms %.3f m\n", base.rms_pos);
  EXPECT_LT(base.rms_pos, 0.05);
}
