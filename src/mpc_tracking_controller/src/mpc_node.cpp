// ROS 2 node (Foxy / Humble): the MPC path-tracking controller on the car's topics.
//
// Same interface and behaviour as the IRL controller node (integral_rl_controller), so the
// two are compared on equal terms:
//   in   odom_topic        nav_msgs/Odometry        localisation pose (particle filter)
//        path_topic        nav_msgs/Path            reference path (latched or volatile)
//        target_speed_topic std_msgs/Float64        speed reference [m/s]
//        imu_topic         sensor_msgs/Imu          yaw rate (latency compensation)
//        vesc_state_topic  vesc_msgs/VescStateStamped  state.speed in ERPM -> speed
//        joy_topic         sensor_msgs/Joy          button `joy_button` toggles autonomous mode
//   out  motor_topic       std_msgs/Float64         v_cmd * speed_cmd_per_mps (ERPM)
//        servo_topic       std_msgs/Float64         servo_center + delta * servo_per_rad (clamped)
//        /mpc/state        std_msgs/Float64MultiArray  [pose stamp, e_lat, e_psi, x0 (4), s0, v_ref,
//                                                   v_cmd, delta_cmd, solve ms, QP iterations, converged,
//                                                   steering offset, grip [g], slipping, understeer]
// One control step per pose message (decimated to control_period).
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <vesc_msgs/msg/vesc_state_stamped.hpp>

#include "mpc_tracking_controller/tracking_mpc.hpp"

using std::placeholders::_1;
using SteadyClock = std::chrono::steady_clock;

namespace mpc_tc {

class MpcNode : public rclcpp::Node {
 public:
  MpcNode() : Node("mpc_controller") {
    auto D = [this](const std::string& n, double v) { return this->declare_parameter<double>(n, v); };
    auto S = [this](const std::string& n, const std::string& v) { return this->declare_parameter<std::string>(n, v); };
    // topics
    const auto odom_topic = S("odom_topic", "/pf/pose/odom");
    const auto path_topic = S("path_topic", "/path");
    const auto speed_topic = S("target_speed_topic", "/target_speed");
    const auto joy_topic = S("joy_topic", "/joy");
    const auto servo_topic = S("servo_topic", "/commands/servo/position");
    const auto motor_topic = S("motor_topic", "/commands/motor/speed");
    const auto imu_topic = S("imu_topic", "/zed2/zed_node/imu/data");
    const auto vesc_topic = S("vesc_state_topic", "/sensors/core");
    // actuator calibration (controller.yaml real.*)
    speed_cmd_per_mps_ = D("speed_cmd_per_mps", 2171.0);
    servo_per_rad_ = D("servo_per_rad", 0.929);
    servo_center_ = D("servo_center", 0.5);
    servo_min_ = D("servo_min", 0.2);
    servo_max_ = D("servo_max", 0.8);
    require_arming_ = this->declare_parameter<bool>("require_arming", true);
    // true: the pose stamp is its measurement time (particle filter scan_synchronous),
    // latency = arrival - stamp + model.pose_latency_* (residual); false: fixed model.pose_latency_*
    pose_latency_from_stamp_ = this->declare_parameter<bool>("pose_latency_from_stamp", false);
    joy_button_ = this->declare_parameter<int>("joy_button", 0);
    joy_cooldown_ = D("joy_cooldown", 0.5);
    odom_timeout_ = D("odom_timeout", 0.5);
    speed_timeout_ = D("target_speed_timeout", 1.0);

    ControllerParams p;
    p.model.wheelbase = D("model.wheelbase", p.model.wheelbase);
    p.model.lever_arm = D("model.lever_arm", p.model.lever_arm);
    p.model.steer_tau = D("model.steer_tau", p.model.steer_tau);
    p.model.speed_tau = D("model.speed_tau", p.model.speed_tau);
    p.model.speed_gain = D("model.speed_gain", p.model.speed_gain);
    p.steer_dead_time = D("model.steer_dead_time", p.steer_dead_time);
    p.speed_dead_time = D("model.speed_dead_time", p.speed_dead_time);
    p.use_steer_map = this->declare_parameter<bool>("model.use_steer_map", p.use_steer_map);
    p.steer_map_cmd = this->declare_parameter<std::vector<double>>("model.steer_map_cmd", p.steer_map_cmd);
    p.steer_map_eff = this->declare_parameter<std::vector<double>>("model.steer_map_eff", p.steer_map_eff);
    p.pose_latency_position = D("model.pose_latency_position", p.pose_latency_position);
    p.pose_latency_heading = D("model.pose_latency_heading", p.pose_latency_heading);
    p.steer_offset_tau = D("steer_offset.tau", p.steer_offset_tau);
    p.grip_adaptation = this->declare_parameter<bool>("grip.adaptation", p.grip_adaptation);
    p.grip_nominal = D("grip.nominal", p.grip_nominal);
    p.grip_min = D("grip.min", p.grip_min);
    p.grip_margin = D("grip.margin", p.grip_margin);
    p.grip_recover_tau = D("grip.recover_tau", p.grip_recover_tau);
    p.mpc.drive_axle_load = D("grip.drive_axle_load", p.mpc.drive_axle_load);
    p.slip_rel = D("grip.slip_rel", p.slip_rel);
    p.slip_abs = D("grip.slip_abs", p.slip_abs);
    p.slip_min_speed = D("grip.slip_min_speed", p.slip_min_speed);
    p.model.understeer = D("model.understeer", p.model.understeer);
    p.understeer_tau = D("understeer.tau", p.understeer_tau);
    p.understeer_max = D("understeer.max", p.understeer_max);
    p.steer_offset_max = D("steer_offset.max", p.steer_offset_max);
    p.steer_offset_min_speed = D("steer_offset.min_speed", p.steer_offset_min_speed);
    p.steer_offset_max_delta = D("steer_offset.max_delta", p.steer_offset_max_delta);
    p.heading_bias = D("model.heading_bias", p.heading_bias);
    p.compute_delay = D("model.compute_delay", p.compute_delay);
    p.use_imu = this->declare_parameter<bool>("use_imu", p.use_imu);
    p.delta_cmd_max = D("delta_max", p.delta_cmd_max);
    p.control_period = D("control_period", p.control_period);
    p.abort_e_lat = D("abort_e_lat", p.abort_e_lat);
    p.abort_e_psi = D("abort_e_psi", p.abort_e_psi);
    auto& m = p.mpc;
    m.horizon = this->declare_parameter<int>("mpc.horizon", m.horizon);
    m.dt = D("mpc.dt", m.dt);
    m.substeps = this->declare_parameter<int>("mpc.substeps", m.substeps);
    m.max_e_lat = D("mpc.max_e_lat", m.max_e_lat);
    m.max_e_psi = D("mpc.max_e_psi", m.max_e_psi);
    m.max_e_v = D("mpc.max_e_v", m.max_e_v);
    m.max_v_rate = D("mpc.max_v_rate", m.max_v_rate);
    m.max_delta_rate = D("mpc.max_delta_rate", m.max_delta_rate);
    m.terminal_factor = D("mpc.terminal_factor", m.terminal_factor);
    m.sqp_iterations = this->declare_parameter<int>("mpc.sqp_iterations", m.sqp_iterations);
    m.cap_preview_stages = this->declare_parameter<int>("mpc.cap_preview_stages", m.cap_preview_stages);
    m.qp.max_iter = this->declare_parameter<int>("mpc.qp_max_iter", m.qp.max_iter);
    m.qp.eps_abs = D("mpc.qp_eps_abs", m.qp.eps_abs);
    m.qp.eps_rel = D("mpc.qp_eps_rel", m.qp.eps_rel);
    m.v_min = D("v_min", m.v_min);
    m.v_max = D("v_max", m.v_max);
    m.v_rate_max = D("v_rate_max", m.v_rate_max);
    m.delta_rate_max = D("delta_rate_max", m.delta_rate_max);
    m.a_lat_max = D("a_lat_max", m.a_lat_max);
    // the speed channel's dead time beyond the steering's, as whole MPC stages
    m.speed_delay_steps = std::max(0, static_cast<int>(std::lround((p.speed_dead_time - p.steer_dead_time) / m.dt)));
    if (p.steer_map_cmd.size() != p.steer_map_eff.size() || p.steer_map_cmd.size() < 2)
      throw std::invalid_argument("model.steer_map_cmd / steer_map_eff: same length >= 2 required");
    {  // effective steering bounds = the steering map of the command bounds
      const TrackingController tmp(p);
      m.delta_lo = tmp.steer_to_eff(-p.delta_cmd_max);
      m.delta_hi = tmp.steer_to_eff(p.delta_cmd_max);
    }
    ctrl_ = std::make_unique<TrackingController>(p);
    control_period_ = p.control_period;

    pub_motor_ = create_publisher<std_msgs::msg::Float64>(motor_topic, 10);
    pub_servo_ = create_publisher<std_msgs::msg::Float64>(servo_topic, 10);
    pub_state_ = create_publisher<std_msgs::msg::Float64MultiArray>("/mpc/state", 10);
    auto sensor_qos = rclcpp::QoS(1).best_effort();
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic, sensor_qos, std::bind(&MpcNode::on_odom, this, _1));
    // the path: latched (transient local) and volatile publishers both work
    sub_path_latched_ = create_subscription<nav_msgs::msg::Path>(
        path_topic, rclcpp::QoS(1).reliable().transient_local(), std::bind(&MpcNode::on_path, this, _1));
    sub_path_ = create_subscription<nav_msgs::msg::Path>(path_topic, 10, std::bind(&MpcNode::on_path, this, _1));
    sub_speed_ = create_subscription<std_msgs::msg::Float64>(speed_topic, 10, [this](std_msgs::msg::Float64::SharedPtr msg) {
      v_ref_ = msg->data;
      v_ref_wall_ = SteadyClock::now();
      have_v_ref_ = true;
    });
    sub_joy_ = create_subscription<sensor_msgs::msg::Joy>(joy_topic, 10, std::bind(&MpcNode::on_joy, this, _1));
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic, sensor_qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
      ctrl_->on_imu(now_s(), msg->angular_velocity.z);
    });
    sub_vesc_ = create_subscription<vesc_msgs::msg::VescStateStamped>(vesc_topic, sensor_qos,
                                                                        [this](vesc_msgs::msg::VescStateStamped::SharedPtr msg) {
      ctrl_->on_speed(now_s(), msg->state.speed / speed_cmd_per_mps_);
      speed_wall_ = SteadyClock::now();
      have_speed_ = true;
    });
    armed_ = !require_arming_;
    watchdog_ = create_wall_timer(std::chrono::milliseconds(100), std::bind(&MpcNode::watchdog, this));
    report_ = create_wall_timer(std::chrono::seconds(5), std::bind(&MpcNode::report, this));
    RCLCPP_INFO(get_logger(),
                "MPC: horizon %d x %.2f s, speed delay %d stages, steering %.3f..%.3f rad (effective), "
                "speed_cmd_per_mps %.1f, servo_per_rad %.3f, arming %s",
                m.horizon, m.dt, m.speed_delay_steps, m.delta_lo, m.delta_hi, speed_cmd_per_mps_, servo_per_rad_,
                require_arming_ ? "joystick" : "off");
  }

 private:
  double now_s() { return this->now().seconds(); }
  static double age(SteadyClock::time_point t) {
    return std::chrono::duration<double>(SteadyClock::now() - t).count();
  }

  void on_path(const nav_msgs::msg::Path::SharedPtr msg) {
    const size_t n = msg->poses.size();
    if (n < 3) return;
    const auto& f = msg->poses.front().pose.position;
    const auto& b = msg->poses.back().pose.position;
    const auto& c = msg->poses[n / 2].pose.position;
    const double sig[7] = {static_cast<double>(n), f.x, f.y, b.x, b.y, c.x, c.y};
    if (ctrl_->has_path() && std::equal(sig, sig + 7, path_sig_)) return;  // a republish of the same path
    std::vector<double> x(n), y(n);
    for (size_t i = 0; i < n; ++i) {
      x[i] = msg->poses[i].pose.position.x;
      y[i] = msg->poses[i].pose.position.y;
    }
    try {
      ctrl_->set_path(x, y);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "path rejected: %s", e.what());
      return;
    }
    std::copy(sig, sig + 7, path_sig_);
    path_frame_ = msg->header.frame_id;
    frames_checked_ = false;
    RCLCPP_INFO(get_logger(), "path: %zu poses, %.1f m, closed=%s", n, ctrl_->path()->length(),
                ctrl_->path()->closed() ? "true" : "false");
  }

  void on_joy(const sensor_msgs::msg::Joy::SharedPtr msg) {
    if (static_cast<int>(msg->buttons.size()) <= joy_button_ || msg->buttons[joy_button_] != 1) return;
    if (age(last_toggle_) < joy_cooldown_) return;
    last_toggle_ = SteadyClock::now();
    armed_ = !armed_;
    ctrl_->reset_mpc();
    RCLCPP_INFO(get_logger(), "autonomous mode: %s", armed_ ? "ON" : "OFF");
  }

  void publish(double v_cmd, double delta_cmd) {
    std_msgs::msg::Float64 mo, se;
    mo.data = v_cmd * speed_cmd_per_mps_;
    se.data = std::min(servo_max_, std::max(servo_min_, servo_center_ + delta_cmd * servo_per_rad_));
    pub_motor_->publish(mo);
    pub_servo_->publish(se);
    ctrl_->on_command_sent(now_s(), v_cmd, delta_cmd);
  }
  void publish_stop() {
    publish(0.0, 0.0);
    ctrl_->reset_mpc();
  }

  void on_odom(const nav_msgs::msg::Odometry::SharedPtr msg) {
    const auto t0 = SteadyClock::now();
    last_odom_wall_ = t0;
    have_odom_ = true;
    const double t_arr = now_s();
    const double stamp = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;
    if (!frames_checked_ && ctrl_->has_path()) {
      frames_checked_ = true;
      if (!msg->header.frame_id.empty() && msg->header.frame_id != path_frame_)
        RCLCPP_ERROR(get_logger(), "path frame '%s' != odometry frame '%s': both must be the same (no TF is used)",
                     path_frame_.c_str(), msg->header.frame_id.c_str());
    }
    if (last_step_ > 0.0 && t_arr - last_step_ < 0.9 * control_period_) return;  // decimate to the control period
    const bool ready = armed_ && ctrl_->has_path() && have_v_ref_ && age(v_ref_wall_) < speed_timeout_ && have_speed_ &&
                       age(speed_wall_) < odom_timeout_;
    if (!ready) {
      last_step_ = t_arr;
      publish_stop();
      return;
    }
    last_step_ = t_arr;
    const auto& q = msg->pose.pose.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    const double v_ref = std::max(v_ref_, 0.0);  // the MPC bounds it by v_max and the lateral-acceleration cap
    const double pose_age = pose_latency_from_stamp_ ? std::clamp(t_arr - stamp, 0.0, 0.5) : 0.0;
    const ControlOutput o =
        ctrl_->on_pose(t_arr, msg->pose.pose.position.x, msg->pose.pose.position.y, yaw, v_ref, pose_age);
    if (o.status != Status::kOk) {
      publish_stop();
      if (o.status == Status::kAbort) {
        RCLCPP_WARN(get_logger(), "left the tracking envelope or MPC failed (e_lat=%.2f m, e_psi=%.2f rad): stop",
                    o.meas.e_lat, o.meas.e_psi);
        if (require_arming_) {
          armed_ = false;
          RCLCPP_WARN(get_logger(), "disarmed -- re-arm with the joystick");
        }
      }
      return;
    }
    publish(o.v_cmd, o.delta_cmd);
    const double ms = 1e3 * age(t0);
    stat_n_++;
    stat_ms_sum_ += ms;
    stat_ms_max_ = std::max(stat_ms_max_, ms);
    stat_it_sum_ += o.mpc.qp_iterations;
    stat_nc_ += o.mpc.qp_converged ? 0 : 1;
    std_msgs::msg::Float64MultiArray st;
    st.data = {stamp, o.meas.e_lat, o.meas.e_psi, o.x0[0], o.x0[1], o.x0[2], o.x0[3], o.s0, v_ref,
               o.v_cmd, o.delta_cmd, ms, static_cast<double>(o.mpc.qp_iterations), o.mpc.qp_converged ? 1.0 : 0.0, ctrl_->steer_offset(),
               ctrl_->grip(), ctrl_->slipping() ? 1.0 : 0.0, ctrl_->understeer()};
    pub_state_->publish(st);
  }

  void watchdog() {
    if (have_odom_ && age(last_odom_wall_) > odom_timeout_) {
      publish_stop();
      have_odom_ = false;
      RCLCPP_WARN(get_logger(), "no odometry for %.1f s: stop", odom_timeout_);
    }
  }

  void report() {
    if (stat_n_ == 0) return;
    RCLCPP_INFO(get_logger(), "control steps %d: step %.2f ms mean, %.2f ms max; QP iterations %.1f mean; not converged %d",
                stat_n_, stat_ms_sum_ / stat_n_, stat_ms_max_, stat_it_sum_ / stat_n_, stat_nc_);
    stat_n_ = stat_nc_ = 0;
    stat_ms_sum_ = stat_ms_max_ = stat_it_sum_ = 0.0;
  }

  std::unique_ptr<TrackingController> ctrl_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_motor_, pub_servo_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_state_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_latched_, sub_path_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_speed_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_joy_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<vesc_msgs::msg::VescStateStamped>::SharedPtr sub_vesc_;
  rclcpp::TimerBase::SharedPtr watchdog_, report_;
  double speed_cmd_per_mps_, servo_per_rad_, servo_center_, servo_min_, servo_max_;
  bool require_arming_ = true, armed_ = false, pose_latency_from_stamp_ = false;
  int joy_button_ = 0;
  double joy_cooldown_ = 0.5, odom_timeout_ = 0.5, speed_timeout_ = 1.0, control_period_ = 0.1;
  double v_ref_ = 0.0;
  bool have_v_ref_ = false, have_speed_ = false, have_odom_ = false, frames_checked_ = true;
  SteadyClock::time_point v_ref_wall_, speed_wall_, last_odom_wall_, last_toggle_;
  double last_step_ = -1.0;
  double path_sig_[7] = {0, 0, 0, 0, 0, 0, 0};
  std::string path_frame_;
  int stat_n_ = 0, stat_nc_ = 0;
  double stat_ms_sum_ = 0.0, stat_ms_max_ = 0.0, stat_it_sum_ = 0.0;
};

}  // namespace mpc_tc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mpc_tc::MpcNode>());
  rclcpp::shutdown();
  return 0;
}
