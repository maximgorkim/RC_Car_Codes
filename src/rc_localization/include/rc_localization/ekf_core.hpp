// EKF core for the RC car, without any ROS dependency (unit-testable).
//
// State (11):
//   x, y        [m]     position of base_link in the world frame
//   yaw         [rad]   heading of base_link (what the LiDAR pose measures)
//   yaw_bias    [rad]   direction of travel - yaw (slip angle / LiDAR heading
//                       offset), the same idea as Autoware's ekf_localizer
//   vx          [m/s]   forward speed of the rear axle
//   wz          [rad/s] yaw rate
//   gyro_bias   [rad/s] IMU gyro z bias:          wz_meas = wz + gyro_bias
//   speed_scale [-]     VESC speed scale factor:  v_meas  = speed_scale * vx
//   vy          [m/s]   lateral velocity of base_link   (lateral_dynamics only)
//   ay          [m/s^2] lateral acceleration            (lateral_dynamics only)
//   ay_bias     [m/s^2] IMU accel y bias / mounting tilt: ay_meas = ay + ay_bias
//
// Motion model (constant speed / yaw rate). base_link sits lever_arm (a)
// metres ahead of the rear axle. Its lateral velocity v_lat is
//   kinematic (lateral_dynamics = false):  v_lat = a * wz   (wheels never slip)
//   lateral_dynamics = true:               v_lat = vy, a state driven by the
//     measured lateral acceleration (rigid body, body frame):
//       vy' = vy + (ay - vx wz - (vy - a wz) / slip_tau) dt
//     The last term slowly pulls vy back to the no-slip value (slip_tau, a
//     few s) to bound the drift of integrating a noisy accelerometer; the
//     slide itself is in the measured ay. A slide shows up as ay != vx wz.
//   th   = yaw + yaw_bias
//   x'   = x + (vx cos th - v_lat sin th) dt
//   y'   = y + (vx sin th + v_lat cos th) dt
//   yaw' = yaw + wz dt            (all other states: random walk)
//
// Measurements:
//   pose  z = [x, y, yaw]                          LiDAR pose, ~8 Hz, delayed
//   twist z = [speed_scale * vx, wz + gyro_bias]   VESC + IMU, ~190 Hz
//   accel z = ay + ay_bias                         IMU accel y, ~190 Hz (lateral_dynamics)
//
// DelayedEkf keeps a short history of (measurement, state after it). A
// measurement that arrives late is inserted at its own timestamp and every
// later measurement is replayed, so a delayed LiDAR pose is applied at the time
// it was measured (delay compensation) and the latest state stays current.

#pragma once

#include <deque>

#include <Eigen/Dense>

namespace rc_localization
{

constexpr int NX = 11;
enum StateIdx { X = 0, Y, YAW, YAW_BIAS, VX, WZ, GYRO_BIAS, SPEED_SCALE, VY, AY, AY_BIAS };

using StateVec = Eigen::Matrix<double, NX, 1>;
using StateMat = Eigen::Matrix<double, NX, NX>;

double normalize_angle(double a);

struct EkfParams
{
  double lever_arm = 0.17;  // m, base_link ahead of the rear axle

  // Process noise as stddev per sqrt(second): Q_ii = q_i^2 * dt
  double q_pos = 0.02;
  double q_yaw = 0.005;
  double q_yaw_bias = 0.002;
  double q_vx = 4.0;            // ~ longitudinal acceleration [m/s^2]
  double q_wz = 6.0;            // ~ yaw acceleration [rad/s^2]
  double q_gyro_bias = 0.0005;
  double q_speed_scale = 0.002;

  // Lateral dynamics (side slip). Off: the old kinematic model, the three
  // lateral states are not used (and stay at their initial values).
  bool lateral_dynamics = false;
  // s. The measured ay already contains the tyre forces that end a slide, so
  // this is NOT the physical slip decay: it only bounds the drift of
  // integrating a noisy/biased accelerometer. Too short damps real slides
  // (0.3 s: slide error 0.22 m vs 0.05 m with 2 s in the unit test).
  double slip_tau = 2.0;
  double q_vy = 0.05;           // m/s / sqrt(s), unmodelled lateral velocity (0.2: bag jumps up to 18 cm)
  double q_ay = 30.0;           // m/s^2 / sqrt(s), lateral acceleration changes fast
  double q_ay_bias = 0.01;      // m/s^2 / sqrt(s)

  double max_predict_dt = 0.01;  // s, longer predictions are split into steps
  double pose_gate = 16.27;      // squared Mahalanobis, chi2(3 dof) 99.9 %
  double twist_gate = 0.0;       // squared Mahalanobis, chi2(2 dof); 0 = off

  double max_yaw_bias = 0.2;     // rad
  double max_gyro_bias = 0.1;    // rad/s
  double min_speed_scale = 0.7;
  double max_speed_scale = 1.3;
  double max_ay_bias = 2.0;      // m/s^2 (~12 deg of mounting tilt)
};

struct PoseMeas
{
  Eigen::Vector3d z = Eigen::Vector3d::Zero();   // x, y, yaw
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  // Uncertainty of the measurement time [s]. A pose whose timestamp is off by
  // dt is wrong by (velocity * dt) along the motion, so R is extended with
  // time_stddev^2 * u u^T, u = [dx/dt, dy/dt, dyaw/dt] from the filter state.
  double time_stddev = 0.0;
};

struct TwistMeas
{
  Eigen::Vector2d z = Eigen::Vector2d::Zero();   // measured speed, measured yaw rate
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
};

struct AccelMeas
{
  double z = 0.0;   // measured lateral acceleration (IMU y, body frame)
  double R = 1.0;   // variance
};

class Ekf
{
public:
  explicit Ekf(const EkfParams & p) : p_(p) { x.setZero(); P.setIdentity(); }

  void predict(double dt);
  // Both return false (state untouched) if the measurement fails its gate.
  // d2 receives the squared Mahalanobis distance of the innovation.
  bool update_pose(const PoseMeas & m, double * d2 = nullptr);
  bool update_twist(const TwistMeas & m, double * d2 = nullptr);
  bool update_accel(const AccelMeas & m, double * d2 = nullptr);
  // Lateral velocity of base_link used by the motion model.
  double lateral_velocity() const;

  StateVec x;
  StateMat P;

private:
  void predict_step(double dt);
  void finish_update();

  EkfParams p_;
};

struct Measurement
{
  enum class Type { INIT, POSE, TWIST, ACCEL };
  double t = 0.0;
  Type type = Type::INIT;
  PoseMeas pose;
  TwistMeas twist;
  AccelMeas accel;
};

class DelayedEkf
{
public:
  enum class Result { APPLIED, REJECTED, TOO_OLD, NOT_INITIALIZED };

  DelayedEkf(const EkfParams & p, double history_length) : p_(p), history_(history_length), ekf_(p) {}

  bool initialized() const { return !hist_.empty(); }
  void reset() { hist_.clear(); }
  void initialize(double t, const StateVec & x0, const StateMat & P0);

  // Inserts m at m.t and replays every later measurement. d2: Mahalanobis
  // distance of m; replayed: number of later measurements re-processed.
  Result add(const Measurement & m, double * d2 = nullptr, int * replayed = nullptr);

  // Latest state predicted to time t; the filter itself is not modified.
  // Prediction is limited to max_extrapolation seconds past the newest
  // measurement (so a dead sensor cannot make the pose run away).
  void predict_to(double t, double max_extrapolation, StateVec & x, StateMat & P) const;

  double time() const { return hist_.back().m.t; }
  double oldest_time() const { return hist_.front().m.t; }
  const StateVec & state() const { return hist_.back().x; }
  const StateMat & covariance() const { return hist_.back().P; }
  // Filter estimate at a past time t (newest entry not after t, predicted to
  // t). This is what a measurement taken at t is compared with.
  void state_at(double t, StateVec & x, StateMat & P) const;

private:
  struct Entry
  {
    Measurement m;
    StateVec x = StateVec::Zero();  // state right after m was processed
    StateMat P = StateMat::Zero();
    bool accepted = true;
    double d2 = 0.0;
  };

  void process(Entry & e, double t_prev);
  void prune();

  EkfParams p_;
  double history_;
  Ekf ekf_;
  std::deque<Entry> hist_;
};

}  // namespace rc_localization
