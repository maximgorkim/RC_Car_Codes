// ekf_localizer: fuses the high-rate twist (VESC + IMU) with the delayed LiDAR
// pose (particle filter) and publishes a pose at a fixed rate with no lag.
//
// Subscribes:
//   twist_topic        geometry_msgs/TwistWithCovarianceStamped  (twist_estimator)
//   pose_topic         geometry_msgs/PoseWithCovarianceStamped   (pose_msg_type: pose_with_cov)
//                      or nav_msgs/Odometry                      (pose_msg_type: odometry, today's /pf/pose/odom)
//   initialpose_topic  geometry_msgs/PoseWithCovarianceStamped   (RViz "2D Pose Estimate")
//   imu_topic          sensor_msgs/Imu  linear_acceleration.y    (model.lateral_dynamics only)
//   vo_topic           nav_msgs/Odometry, visual odometry POSES (ZED); only with vo.enabled (default off). The
//                      camera velocity from them (vo_velocity.hpp) is a measurement: side slip
//                      seen independently of the wheels. Best with model.lateral_dynamics.
// Publishes (at output_rate, predicted to the output time):
//   odom_topic         nav_msgs/Odometry          drop-in replacement for /pf/pose/odom
//   pose_out_topic     geometry_msgs/PoseWithCovarianceStamped
//   debug_topic        std_msgs/Float64MultiArray (see publish_debug)
//   TF world_frame -> base_frame   (replaces the particle filter's TF; disable one of them)

#include <array>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "rc_localization/ekf_core.hpp"
#include "rc_localization/vo_velocity.hpp"

using namespace rc_localization;
using geometry_msgs::msg::PoseWithCovarianceStamped;
using geometry_msgs::msg::TwistWithCovarianceStamped;

namespace
{

double yaw_from_quat(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

geometry_msgs::msg::Quaternion quat_from_yaw(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.z = std::sin(0.5 * yaw);
  q.w = std::cos(0.5 * yaw);
  return q;
}

}  // namespace

class EkfLocalizer : public rclcpp::Node
{
public:
  EkfLocalizer() : Node("ekf_localizer")
  {
    // ---- topics / frames
    const auto twist_topic = declare_parameter<std::string>("twist_topic", "/localization/twist_with_cov");
    const auto imu_topic = declare_parameter<std::string>("imu_topic", "/zed2/zed_node/imu/data");
    const auto pose_topic = declare_parameter<std::string>("pose_topic", "/pf/pose/odom");
    const auto pose_msg_type = declare_parameter<std::string>("pose_msg_type", "odometry");
    const auto initialpose_topic = declare_parameter<std::string>("initialpose_topic", "/initialpose");
    const auto odom_topic = declare_parameter<std::string>("odom_topic", "/localization/odom");
    const auto pose_out_topic = declare_parameter<std::string>("pose_out_topic", "/localization/pose_with_cov");
    const auto debug_topic = declare_parameter<std::string>("debug_topic", "/localization/ekf/debug");
    const auto vo_topic = declare_parameter<std::string>("vo_topic", "/zed2/zed_node/odom");
    const bool vo_enabled = declare_parameter<bool>("vo.enabled", false);
    world_frame_ = declare_parameter<std::string>("world_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);

    // ---- timing
    const double output_rate = declare_parameter<double>("output_rate", 50.0);
    output_period_ = 1.0 / output_rate;
    // timer: fixed-rate wall/ROS timer, output predicted to now (the car).
    // twist: output on twist arrival (rate-limited to output_rate), predicted
    // to max(now, newest measurement). For simulation, where /clock advances in
    // coarse steps (Gazebo: 10 Hz) and a timer on it would fire only that often.
    output_trigger_ = declare_parameter<std::string>("output_trigger", "timer");
    const double history_length = declare_parameter<double>("history_length", 1.0);
    max_extrapolation_ = declare_parameter<double>("max_extrapolation", 0.1);
    // Subtracted from the pose stamp. Today's particle filter stamps with
    // now() although its pose is ~0.2 s old; 0.0 once it stamps with the scan time.
    pose_stamp_offset_ = declare_parameter<double>("pose_stamp_offset", 0.0);
    sensor_timeout_ = declare_parameter<double>("sensor_timeout", 0.5);

    // ---- measurement noise
    use_pose_covariance_ = declare_parameter<bool>("use_pose_covariance", true);
    pose_stddev_xy_ = declare_parameter<double>("pose_stddev_xy", 0.05);
    pose_stddev_yaw_ = declare_parameter<double>("pose_stddev_yaw", 0.02);
    pose_min_stddev_xy_ = declare_parameter<double>("pose_min_stddev_xy", 0.03);
    pose_min_stddev_yaw_ = declare_parameter<double>("pose_min_stddev_yaw", 0.01);
    pose_time_stddev_ = declare_parameter<double>("pose_time_stddev", 0.0);
    // s the LiDAR position is older than its heading (0: same time, the old behaviour)
    pose_position_delay_ = declare_parameter<double>("pose_position_delay", 0.0);
    twist_min_stddev_vx_ = declare_parameter<double>("twist_min_stddev_vx", 0.05);
    twist_min_stddev_wz_ = declare_parameter<double>("twist_min_stddev_wz", 0.01);
    max_consecutive_rejections_ = declare_parameter<int>("max_consecutive_rejections", 15);
    // Output is withheld and the filter restarted above this (sum of x/y variances, m^2).
    max_position_variance_ = declare_parameter<double>("max_position_variance", 4.0);

    // ---- initial state / uncertainty
    init_yaw_bias_ = declare_parameter<double>("init.yaw_bias", 0.0);
    init_gyro_bias_ = declare_parameter<double>("init.gyro_bias", 0.0);
    init_speed_scale_ = declare_parameter<double>("init.speed_scale", 1.0);
    init_stddev_xy_ = declare_parameter<double>("init.stddev_xy", 0.1);
    init_stddev_yaw_ = declare_parameter<double>("init.stddev_yaw", 0.05);
    init_stddev_yaw_bias_ = declare_parameter<double>("init.stddev_yaw_bias", 0.03);
    init_stddev_vx_ = declare_parameter<double>("init.stddev_vx", 1.0);
    init_stddev_wz_ = declare_parameter<double>("init.stddev_wz", 0.5);
    init_stddev_gyro_bias_ = declare_parameter<double>("init.stddev_gyro_bias", 0.01);
    init_stddev_speed_scale_ = declare_parameter<double>("init.stddev_speed_scale", 0.05);
    init_ay_bias_ = declare_parameter<double>("init.ay_bias", 0.0);
    init_stddev_vy_ = declare_parameter<double>("init.stddev_vy", 0.2);
    init_stddev_ay_ = declare_parameter<double>("init.stddev_ay", 2.0);
    init_stddev_ay_bias_ = declare_parameter<double>("init.stddev_ay_bias", 0.5);
    accel_stddev_ = declare_parameter<double>("accel_stddev", 1.0);   // m/s^2, IMU y incl. vibration
    accel_sign_ = declare_parameter<double>("accel_sign", 1.0);       // -1 if the IMU y axis points right
    // The IMU's lateral reading divided by this = body lateral acceleration
    // (mounting position / axis; calibrated on bags as ay_meas vs v*wz).
    accel_scale_ = declare_parameter<double>("accel_scale", 1.0);
    // visual odometry velocity (vo_topic)
    init_vo_scale_ = declare_parameter<double>("init.vo_scale", 1.0);
    init_stddev_vo_scale_ = declare_parameter<double>("init.stddev_vo_scale", 0.05);
    vo_stddev_ = declare_parameter<double>("vo.stddev", 0.15);     // m/s per component (bags: 0.1-0.17)
    const double vo_window = declare_parameter<double>("vo.window", 0.2);
    const double vo_period = declare_parameter<double>("vo.period", 0.1);
    vo_ = std::make_unique<VoVelocity>(vo_window, vo_period);

    // ---- model
    EkfParams p;
    p.lever_arm = declare_parameter<double>("model.lever_arm", p.lever_arm);
    p.q_pos = declare_parameter<double>("process_noise.pos", p.q_pos);
    p.q_yaw = declare_parameter<double>("process_noise.yaw", p.q_yaw);
    p.q_yaw_bias = declare_parameter<double>("process_noise.yaw_bias", p.q_yaw_bias);
    p.q_vx = declare_parameter<double>("process_noise.vx", p.q_vx);
    p.q_wz = declare_parameter<double>("process_noise.wz", p.q_wz);
    p.q_gyro_bias = declare_parameter<double>("process_noise.gyro_bias", p.q_gyro_bias);
    p.q_speed_scale = declare_parameter<double>("process_noise.speed_scale", p.q_speed_scale);
    p.max_predict_dt = declare_parameter<double>("model.max_predict_dt", p.max_predict_dt);
    p.pose_gate = declare_parameter<double>("gate.pose", p.pose_gate);
    p.twist_gate = declare_parameter<double>("gate.twist", p.twist_gate);
    p.max_yaw_bias = declare_parameter<double>("limits.max_yaw_bias", p.max_yaw_bias);
    p.max_gyro_bias = declare_parameter<double>("limits.max_gyro_bias", p.max_gyro_bias);
    p.min_speed_scale = declare_parameter<double>("limits.min_speed_scale", p.min_speed_scale);
    p.max_speed_scale = declare_parameter<double>("limits.max_speed_scale", p.max_speed_scale);
    p.max_ay_bias = declare_parameter<double>("limits.max_ay_bias", p.max_ay_bias);
    p.lateral_dynamics = declare_parameter<bool>("model.lateral_dynamics", p.lateral_dynamics);
    p.slip_tau = declare_parameter<double>("model.slip_tau", p.slip_tau);
    p.q_vy = declare_parameter<double>("process_noise.vy", p.q_vy);
    p.q_ay = declare_parameter<double>("process_noise.ay", p.q_ay);
    p.q_ay_bias = declare_parameter<double>("process_noise.ay_bias", p.q_ay_bias);
    p.q_vo_scale = declare_parameter<double>("process_noise.vo_scale", p.q_vo_scale);
    p.vo_lever = declare_parameter<double>("vo.lever", p.vo_lever);
    p.vo_gate = declare_parameter<double>("gate.vo", p.vo_gate);
    params_ = p;
    ekf_ = std::make_unique<DelayedEkf>(p, history_length);

    // ---- I/O
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
    pub_pose_ = create_publisher<PoseWithCovarianceStamped>(pose_out_topic, 10);
    pub_debug_ = create_publisher<std_msgs::msg::Float64MultiArray>(debug_topic, 10);
    if (publish_tf_) tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    if (p.lateral_dynamics) {
      sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Imu::SharedPtr m) { on_imu(*m); });
    }
    if (vo_enabled) {
      sub_vo_ = create_subscription<nav_msgs::msg::Odometry>(
        vo_topic, rclcpp::SensorDataQoS(), [this](nav_msgs::msg::Odometry::SharedPtr m) { on_vo(*m); });
    }
    sub_twist_ = create_subscription<TwistWithCovarianceStamped>(
      twist_topic, rclcpp::SensorDataQoS(), [this](TwistWithCovarianceStamped::SharedPtr m) { on_twist(*m); });
    if (pose_msg_type == "odometry") {
      sub_pose_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        pose_topic, rclcpp::SensorDataQoS(), [this](nav_msgs::msg::Odometry::SharedPtr m) {
          on_pose(m->header.stamp, m->pose);
        });
    } else if (pose_msg_type == "pose_with_cov") {
      sub_pose_ = create_subscription<PoseWithCovarianceStamped>(
        pose_topic, rclcpp::SensorDataQoS(), [this](PoseWithCovarianceStamped::SharedPtr m) {
          on_pose(m->header.stamp, m->pose);
        });
    } else {
      throw std::invalid_argument("pose_msg_type must be 'odometry' or 'pose_with_cov'");
    }
    sub_init_ = create_subscription<PoseWithCovarianceStamped>(
      initialpose_topic, 10, [this](PoseWithCovarianceStamped::SharedPtr m) { on_initialpose(*m); });

    // Timer on the node clock, so it follows /clock with use_sim_time (bags).
    if (output_trigger_ == "timer") {
      const rclcpp::Duration period(std::chrono::nanoseconds(static_cast<int64_t>(1e9 / output_rate)));
      timer_ = rclcpp::create_timer(this, get_clock(), period, [this]() { on_timer(); });
    } else if (output_trigger_ != "twist") {
      throw std::invalid_argument("output_trigger must be 'timer' or 'twist'");
    }

    RCLCPP_INFO(get_logger(), "ekf_localizer: twist %s + pose %s (%s, stamp offset %.3f s) -> %s @ %.0f Hz, TF %s->%s %s",
                twist_topic.c_str(), pose_topic.c_str(), pose_msg_type.c_str(), pose_stamp_offset_,
                odom_topic.c_str(), output_rate, world_frame_.c_str(), base_frame_.c_str(),
                publish_tf_ ? "on" : "off");
  }

private:
  double to_sec(const builtin_interfaces::msg::Time & t) const { return t.sec + 1e-9 * t.nanosec; }
  double now_sec() { return get_clock()->now().seconds(); }

  void initialize_at(double t, double x, double y, double yaw)
  {
    StateVec x0 = StateVec::Zero();
    x0(X) = x;
    x0(Y) = y;
    x0(YAW) = yaw;
    x0(YAW_BIAS) = init_yaw_bias_;
    x0(GYRO_BIAS) = init_gyro_bias_;
    x0(SPEED_SCALE) = init_speed_scale_;
    x0(AY_BIAS) = init_ay_bias_;
    x0(VO_SCALE) = init_vo_scale_;
    // Keep the learned biases across a re-initialisation.
    if (ekf_->initialized()) {
      x0(AY_BIAS) = ekf_->state()(AY_BIAS);
      x0(VO_SCALE) = ekf_->state()(VO_SCALE);
      x0(YAW_BIAS) = ekf_->state()(YAW_BIAS);
      x0(GYRO_BIAS) = ekf_->state()(GYRO_BIAS);
      x0(SPEED_SCALE) = ekf_->state()(SPEED_SCALE);
      x0(VX) = ekf_->state()(VX);
      x0(WZ) = ekf_->state()(WZ);
    }
    StateVec sd;
    sd << init_stddev_xy_, init_stddev_xy_, init_stddev_yaw_, init_stddev_yaw_bias_, init_stddev_vx_,
      init_stddev_wz_, init_stddev_gyro_bias_, init_stddev_speed_scale_, init_stddev_vy_, init_stddev_ay_,
      init_stddev_ay_bias_, init_stddev_vo_scale_;
    const StateMat P0 = sd.array().square().matrix().asDiagonal();
    ekf_->initialize(t, x0, P0);
    consecutive_rejections_ = 0;
  }

  // A NaN / inf measurement would corrupt the filter for good: drop it.
  bool finite_or_warn(std::initializer_list<double> values, const char * what)
  {
    for (double v : values) {
      if (!std::isfinite(v)) {
        ++dropped_non_finite_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "non-finite %s dropped (%ld so far)", what,
                             dropped_non_finite_);
        return false;
      }
    }
    return true;
  }

  void on_vo(const nav_msgs::msg::Odometry & msg)
  {
    const auto & p = msg.pose.pose;
    if (!finite_or_warn({p.position.x, p.position.y, p.orientation.z, p.orientation.w}, "VO pose")) return;
    const auto & q = p.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    VoVelocity::Out o;
    if (!vo_->add(to_sec(msg.header.stamp), p.position.x, p.position.y, yaw, o) || !ekf_->initialized()) return;
    Measurement m;
    m.t = o.t;
    m.type = Measurement::Type::VO;
    m.vo.z << o.v_fwd, o.v_left;
    m.vo.R = (vo_stddev_ * vo_stddev_) * Eigen::Matrix2d::Identity();
    double d2 = 0.0;
    if (ekf_->add(m, &d2) == DelayedEkf::Result::REJECTED) ++vo_rejections_;
    last_vo_d2_ = d2;
  }

  void on_imu(const sensor_msgs::msg::Imu & msg)
  {
    if (!ekf_->initialized()) return;
    if (!finite_or_warn({msg.linear_acceleration.y}, "IMU acceleration")) return;
    Measurement m;
    m.t = to_sec(msg.header.stamp);
    m.type = Measurement::Type::ACCEL;
    m.accel.z = accel_sign_ * msg.linear_acceleration.y / accel_scale_;
    m.accel.R = accel_stddev_ * accel_stddev_;
    ekf_->add(m);
  }

  void on_twist(const TwistWithCovarianceStamped & msg)
  {
    last_twist_rx_ = now_sec();
    if (!ekf_->initialized()) return;
    if (!finite_or_warn({msg.twist.twist.linear.x, msg.twist.twist.angular.z, msg.twist.covariance[0],
                         msg.twist.covariance[35]}, "twist")) return;
    Measurement m;
    m.t = to_sec(msg.header.stamp);
    m.type = Measurement::Type::TWIST;
    m.twist.z << msg.twist.twist.linear.x, msg.twist.twist.angular.z;
    const double var_v = std::max(msg.twist.covariance[0], twist_min_stddev_vx_ * twist_min_stddev_vx_);
    const double var_w = std::max(msg.twist.covariance[35], twist_min_stddev_wz_ * twist_min_stddev_wz_);
    m.twist.R = Eigen::Vector2d(var_v, var_w).asDiagonal();

    const auto r = ekf_->add(m);
    if (r == DelayedEkf::Result::TOO_OLD) {
      ++twist_too_old_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "twist older than the EKF history, dropped");
    }
    if (output_trigger_ == "twist" && ekf_->time() - last_output_t_ >= 0.999 * output_period_) {
      publish_output(std::max(now_sec(), ekf_->time()));
    }
  }

  void on_pose(const builtin_interfaces::msg::Time & stamp, const geometry_msgs::msg::PoseWithCovariance & pose)
  {
    last_pose_rx_ = now_sec();
    const double t = to_sec(stamp) - pose_stamp_offset_;
    const double px = pose.pose.position.x, py = pose.pose.position.y;
    const double pyaw = yaw_from_quat(pose.pose.orientation);
    if (!finite_or_warn({t, px, py, pyaw, pose.covariance[0], pose.covariance[7], pose.covariance[35]},
                        "LiDAR pose")) return;

    // First pose (or a clock jump backwards, e.g. a looping bag): (re)start here.
    if (!ekf_->initialized() || t < ekf_->oldest_time() - 2.0) {
      initialize_at(t, px, py, pyaw);
      RCLCPP_INFO(get_logger(), "initialized at (%.2f, %.2f, %.2f) from the LiDAR pose", px, py, pyaw);
      return;
    }

    Measurement m;
    m.t = t;
    m.type = Measurement::Type::POSE;
    m.pose.z << px, py, pyaw;
    double var_xy = pose_stddev_xy_ * pose_stddev_xy_;
    double var_yaw = pose_stddev_yaw_ * pose_stddev_yaw_;
    if (use_pose_covariance_) {
      var_xy = std::max(std::max(pose.covariance[0], pose.covariance[7]), pose_min_stddev_xy_ * pose_min_stddev_xy_);
      var_yaw = std::max(pose.covariance[35], pose_min_stddev_yaw_ * pose_min_stddev_yaw_);
    }
    m.pose.R = Eigen::Vector3d(var_xy, var_xy, var_yaw).asDiagonal();
    m.pose.time_stddev = pose_time_stddev_;
    m.pose.position_delay = pose_position_delay_;

    double d2 = 0.0;
    int replayed = 0;
    const auto r = ekf_->add(m, &d2, &replayed);
    last_pose_d2_ = d2;
    last_replayed_ = replayed;
    last_pose_delay_ = ekf_->time() - t;

    switch (r) {
      case DelayedEkf::Result::APPLIED:
        consecutive_rejections_ = 0;
        break;
      case DelayedEkf::Result::REJECTED:
        ++total_rejections_;
        ++consecutive_rejections_;
        RCLCPP_WARN(get_logger(), "LiDAR pose rejected (Mahalanobis^2 %.1f > %.1f), %d in a row",
                    d2, params_.pose_gate, consecutive_rejections_);
        if (consecutive_rejections_ >= max_consecutive_rejections_) {
          // The EKF disagrees with the LiDAR for too long (kidnapped / diverged):
          // trust the LiDAR and restart from it.
          RCLCPP_ERROR(get_logger(), "%d LiDAR poses rejected in a row, re-initializing from the LiDAR pose",
                       consecutive_rejections_);
          initialize_at(t, px, py, pyaw);
          ++resets_;
        }
        break;
      case DelayedEkf::Result::TOO_OLD:
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "LiDAR pose is older than the EKF history (%.2f s), dropped; raise history_length",
                             ekf_->time() - t);
        break;
      case DelayedEkf::Result::NOT_INITIALIZED:
        break;
    }
  }

  void on_initialpose(const PoseWithCovarianceStamped & msg)
  {
    const double t = ekf_->initialized() ? ekf_->time() : now_sec();
    const double yaw = yaw_from_quat(msg.pose.pose.orientation);
    initialize_at(t, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw);
    RCLCPP_INFO(get_logger(), "initialized at (%.2f, %.2f, %.2f) from the initial pose",
                msg.pose.pose.position.x, msg.pose.pose.position.y, yaw);
  }

  void on_timer()
  {
    if (!ekf_->initialized()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000, "waiting for the first LiDAR pose");
      return;
    }
    publish_output(now_sec());
  }

  void publish_output(double t_now)
  {
    last_output_t_ = t_now;
    if (t_now - last_twist_rx_ > sensor_timeout_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "no twist for %.2f s (IMU/VESC down?)",
                           t_now - last_twist_rx_);
    }
    if (t_now - last_pose_rx_ > 2.0 * sensor_timeout_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "no LiDAR pose for %.2f s, dead reckoning only",
                           t_now - last_pose_rx_);
    }

    StateVec x;
    StateMat P;
    ekf_->predict_to(t_now, max_extrapolation_, x, P);
    // Divergence guard: never publish garbage; restart from the next LiDAR pose.
    if (!x.allFinite() || !P.allFinite() || P(X, X) + P(Y, Y) > max_position_variance_) {
      RCLCPP_ERROR(get_logger(), "EKF diverged (position variance %.2f m^2), waiting for the next LiDAR pose",
                   P(X, X) + P(Y, Y));
      ekf_->reset();
      ++resets_;
      return;
    }
    const rclcpp::Time stamp(static_cast<int64_t>(std::llround(t_now * 1e9)), get_clock()->get_clock_type());

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = world_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = x(X);
    odom.pose.pose.position.y = x(Y);
    odom.pose.pose.orientation = quat_from_yaw(x(YAW));
    auto & pc = odom.pose.covariance;
    pc.fill(0.0);
    pc[0] = P(X, X);    pc[1] = P(X, Y);    pc[5] = P(X, YAW);
    pc[6] = P(Y, X);    pc[7] = P(Y, Y);    pc[11] = P(Y, YAW);
    pc[30] = P(YAW, X); pc[31] = P(YAW, Y); pc[35] = P(YAW, YAW);
    pc[14] = pc[21] = pc[28] = 1e-6;
    // Twist in base_link: forward speed (rear axle), lateral speed of base_link.
    odom.twist.twist.linear.x = x(VX);
    odom.twist.twist.linear.y = params_.lateral_dynamics ? x(VY) : params_.lever_arm * x(WZ);
    odom.twist.twist.angular.z = x(WZ);
    auto & tc = odom.twist.covariance;
    tc.fill(0.0);
    tc[0] = P(VX, VX);
    tc[35] = P(WZ, WZ);
    pub_odom_->publish(odom);

    PoseWithCovarianceStamped pose;
    pose.header = odom.header;
    pose.pose = odom.pose;
    pub_pose_->publish(pose);

    if (tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = x(X);
      tf.transform.translation.y = x(Y);
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_->sendTransform(tf);
    }

    publish_debug(t_now);
  }

  // [0] newest measurement age (now - filter time) [s]
  // [1] last LiDAR pose delay (filter time - pose time) [s]
  // [2] last LiDAR pose Mahalanobis^2   [3] measurements replayed for it
  // [4] gyro_bias [rad/s]  [5] speed_scale  [6] yaw_bias [rad]
  // [7] total rejected poses  [8] resets  [9] pos stddev [m]  [10] yaw stddev [rad]
  // [11] slip velocity vy - lever_arm*wz [m/s] (0 without lateral_dynamics)  [12] ay_bias [m/s^2]
  // [13] vo_scale  [14] last VO Mahalanobis^2  [15] rejected VO velocities (vo_topic only)
  void publish_debug(double t_now)
  {
    const StateVec & x = ekf_->state();
    const StateMat & P = ekf_->covariance();
    std_msgs::msg::Float64MultiArray d;
    d.data = {t_now - ekf_->time(), last_pose_delay_, last_pose_d2_, static_cast<double>(last_replayed_),
              x(GYRO_BIAS), x(SPEED_SCALE), x(YAW_BIAS), static_cast<double>(total_rejections_),
              static_cast<double>(resets_), std::sqrt(0.5 * (P(X, X) + P(Y, Y))), std::sqrt(P(YAW, YAW)),
              params_.lateral_dynamics ? x(VY) - params_.lever_arm * x(WZ) : 0.0, x(AY_BIAS), x(VO_SCALE),
              last_vo_d2_, static_cast<double>(vo_rejections_)};
    pub_debug_->publish(d);
  }

  // parameters
  std::string world_frame_, base_frame_, output_trigger_;
  double output_period_ = 0.02, last_output_t_ = 0.0;
  bool publish_tf_;
  double max_extrapolation_, pose_stamp_offset_, sensor_timeout_;
  bool use_pose_covariance_;
  double pose_stddev_xy_, pose_stddev_yaw_, pose_min_stddev_xy_, pose_min_stddev_yaw_, pose_time_stddev_;
  double pose_position_delay_;
  double twist_min_stddev_vx_, twist_min_stddev_wz_;
  int max_consecutive_rejections_;
  double max_position_variance_;
  double init_yaw_bias_, init_gyro_bias_, init_speed_scale_;
  double init_stddev_xy_, init_stddev_yaw_, init_stddev_yaw_bias_, init_stddev_vx_, init_stddev_wz_;
  double init_stddev_gyro_bias_, init_stddev_speed_scale_;
  double init_ay_bias_, init_stddev_vy_, init_stddev_ay_, init_stddev_ay_bias_, accel_stddev_, accel_sign_,
    accel_scale_;
  double init_vo_scale_, init_stddev_vo_scale_, vo_stddev_;
  EkfParams params_;

  // state
  std::unique_ptr<DelayedEkf> ekf_;
  double last_twist_rx_ = 0.0, last_pose_rx_ = 0.0;
  double last_pose_d2_ = 0.0, last_pose_delay_ = 0.0;
  int last_replayed_ = 0, consecutive_rejections_ = 0, total_rejections_ = 0, resets_ = 0;
  long twist_too_old_ = 0, dropped_non_finite_ = 0;
  std::unique_ptr<VoVelocity> vo_;
  double last_vo_d2_ = 0.0;
  long vo_rejections_ = 0;

  // ROS
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_debug_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<TwistWithCovarianceStamped>::SharedPtr sub_twist_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_vo_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose_odom_;
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr sub_pose_;
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr sub_init_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EkfLocalizer>());
  rclcpp::shutdown();
  return 0;
}
