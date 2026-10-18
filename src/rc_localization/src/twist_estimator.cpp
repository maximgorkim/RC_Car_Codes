// twist_estimator: VESC wheel speed + IMU yaw rate -> TwistWithCovarianceStamped
//
// RC-car equivalent of Autoware's gyro_odometer. It produces the high-rate
// "fast" input of the EKF: forward speed from the VESC ERPM and yaw rate from
// the IMU gyro (not from the steering angle, which slips at high speed).
//
// One twist is published per IMU message, stamped with the IMU's own
// timestamp, using the latest VESC speed (if it is not stale).
//
// Subscribes:
//   vesc_state_topic  vesc_msgs/VescStateStamped   state.speed = ERPM
//   imu_topic         sensor_msgs/Imu              angular_velocity.z
// Publishes:
//   twist_topic       geometry_msgs/TwistWithCovarianceStamped  (frame base_link)

#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <vesc_msgs/msg/vesc_state_stamped.hpp>

class TwistEstimator : public rclcpp::Node
{
public:
  TwistEstimator() : Node("twist_estimator")
  {
    const auto vesc_topic = declare_parameter<std::string>("vesc_state_topic", "/sensors/core");
    const auto imu_topic = declare_parameter<std::string>("imu_topic", "/zed2/zed_node/imu/data");
    const auto twist_topic = declare_parameter<std::string>("twist_topic", "/localization/twist_with_cov");
    frame_id_ = declare_parameter<std::string>("frame_id", "base_link");

    // 1 m/s = 60 * (poles/2) * gear / (2 pi r) = 3365.4 ERPM (race_car_parameters.txt).
    // The EKF also estimates a scale factor, so a small error here is tolerated.
    erpm_per_mps_ = declare_parameter<double>("erpm_per_mps", 3365.4);
    speed_sign_ = declare_parameter<double>("speed_sign", 1.0);       // -1 if ERPM is negative going forward
    yaw_rate_sign_ = declare_parameter<double>("yaw_rate_sign", 1.0); // -1 if the IMU is mounted upside down
    vesc_timeout_ = declare_parameter<double>("vesc_timeout", 0.1);   // s
    standstill_erpm_ = declare_parameter<double>("standstill_erpm", 50.0);

    // Standard deviations written into the covariance (the EKF uses them as R).
    speed_stddev_ = declare_parameter<double>("speed_stddev", 0.1);         // m/s
    yaw_rate_stddev_ = declare_parameter<double>("yaw_rate_stddev", 0.02);  // rad/s

    pub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(twist_topic, 10);
    sub_vesc_ = create_subscription<vesc_msgs::msg::VescStateStamped>(
      vesc_topic, rclcpp::SensorDataQoS(),
      std::bind(&TwistEstimator::on_vesc, this, std::placeholders::_1));
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::SensorDataQoS(),
      std::bind(&TwistEstimator::on_imu, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "twist_estimator: %s + %s -> %s (%.1f ERPM per m/s)",
                vesc_topic.c_str(), imu_topic.c_str(), twist_topic.c_str(), erpm_per_mps_);
  }

private:
  void on_vesc(const vesc_msgs::msg::VescStateStamped::SharedPtr msg)
  {
    const double erpm = msg->state.speed;
    speed_ = std::abs(erpm) < standstill_erpm_ ? 0.0 : speed_sign_ * erpm / erpm_per_mps_;
    last_vesc_stamp_ = rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type());
    have_vesc_ = true;
  }

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!have_vesc_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "waiting for VESC state");
      return;
    }
    const rclcpp::Time stamp(msg->header.stamp, get_clock()->get_clock_type());
    const double vesc_age = (stamp - last_vesc_stamp_).seconds();
    if (std::abs(vesc_age) > vesc_timeout_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "VESC speed is %.3f s away from the IMU stamp, twist not published", vesc_age);
      return;
    }

    geometry_msgs::msg::TwistWithCovarianceStamped out;
    out.header.stamp = msg->header.stamp;  // measurement time, never now()
    out.header.frame_id = frame_id_;
    out.twist.twist.linear.x = speed_;
    out.twist.twist.angular.z = yaw_rate_sign_ * msg->angular_velocity.z;

    // 6x6 row-major [vx vy vz wx wy wz]; unused axes get a large variance.
    auto & c = out.twist.covariance;
    c.fill(0.0);
    c[0] = speed_stddev_ * speed_stddev_;
    c[7] = c[14] = c[21] = c[28] = 1.0e4;
    c[35] = yaw_rate_stddev_ * yaw_rate_stddev_;

    pub_->publish(out);
  }

  std::string frame_id_;
  double erpm_per_mps_, speed_sign_, yaw_rate_sign_, vesc_timeout_, standstill_erpm_;
  double speed_stddev_, yaw_rate_stddev_;

  double speed_ = 0.0;
  bool have_vesc_ = false;
  rclcpp::Time last_vesc_stamp_;

  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr pub_;
  rclcpp::Subscription<vesc_msgs::msg::VescStateStamped>::SharedPtr sub_vesc_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TwistEstimator>());
  rclcpp::shutdown();
  return 0;
}
