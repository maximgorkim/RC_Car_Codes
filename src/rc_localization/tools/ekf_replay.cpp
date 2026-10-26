// ekf_replay: runs the EKF offline on a recorded sensor stream, exactly like
// twist_estimator + ekf_localizer would, and reports how well it explains the
// LiDAR poses. Used to tune the parameters on bags in seconds.
//
//   ekf_replay events.csv [key=value ...] [--out trajectory.csv]
//
// events.csv rows, in arrival (bag receive) order:
//   imu,<recv>,<stamp>,<wz>[,<ay>]
//   vesc,<recv>,<stamp>,<erpm>
//   pose,<recv>,<stamp>,<x>,<y>,<yaw>,<var_x>,<var_y>,<var_yaw>
//
// Keys: erpm_per_mps pose_stamp_offset lever_arm q_pos q_yaw q_yaw_bias q_vx
//   q_wz q_gyro_bias q_speed_scale pose_gate pose_min_xy pose_min_yaw
//   speed_std yaw_rate_std init_yaw_bias init_gyro_bias init_speed_scale history
//   output_rate pose_time_std pose_position_delay lateral slip_tau q_vy q_ay q_ay_bias accel_std accel_scale init_ay_bias

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rc_localization/ekf_core.hpp"
#include "rc_localization/vo_velocity.hpp"

using namespace rc_localization;

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s events.csv [key=value ...] [--out traj.csv]\n", argv[0]);
    return 1;
  }
  std::map<std::string, double> k = {
    {"erpm_per_mps", 2080.0}, {"pose_stamp_offset", 0.20}, {"pose_min_xy", 0.03}, {"pose_min_yaw", 0.015},
    {"speed_std", 0.1}, {"yaw_rate_std", 0.02}, {"init_gyro_bias", 0.0}, {"init_speed_scale", 1.0},
    {"history", 1.0}, {"output_rate", 50.0}, {"standstill_erpm", 50.0}, {"max_rejections", 15},
    {"warmup", 5.0}, {"pose_time_std", 0.04}, {"pose_position_delay", 0.0}, {"init_yaw_bias", 0.05},
    {"lateral", 0.0}, {"accel_std", 1.0}, {"accel_scale", 1.0}, {"init_ay_bias", 0.0},
    {"vo", 0.0}, {"vo_std", 0.15}, {"vo_window", 0.2}, {"vo_period", 0.1}, {"init_vo_scale", 1.04}};
  EkfParams p;
  std::map<std::string, double *> pk = {
    {"lever_arm", &p.lever_arm}, {"q_pos", &p.q_pos}, {"q_yaw", &p.q_yaw}, {"q_yaw_bias", &p.q_yaw_bias},
    {"q_vx", &p.q_vx}, {"q_wz", &p.q_wz}, {"q_gyro_bias", &p.q_gyro_bias}, {"q_speed_scale", &p.q_speed_scale},
    {"pose_gate", &p.pose_gate}, {"max_yaw_bias", &p.max_yaw_bias}, {"slip_tau", &p.slip_tau},
    {"q_vy", &p.q_vy}, {"q_ay", &p.q_ay}, {"q_ay_bias", &p.q_ay_bias}, {"q_vo_scale", &p.q_vo_scale},
    {"vo_lever", &p.vo_lever}, {"vo_gate", &p.vo_gate}};
  std::string out_path, innov_path;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) { out_path = argv[++i]; continue; }
    if (a == "--innov" && i + 1 < argc) { innov_path = argv[++i]; continue; }
    const auto eq = a.find('=');
    if (eq == std::string::npos) { std::fprintf(stderr, "bad argument %s\n", a.c_str()); return 1; }
    const std::string key = a.substr(0, eq);
    const double val = std::stod(a.substr(eq + 1));
    if (pk.count(key)) *pk[key] = val;
    else if (k.count(key)) k[key] = val;
    else { std::fprintf(stderr, "unknown key %s\n", key.c_str()); return 1; }
  }

  p.lateral_dynamics = k["lateral"] > 0.5;
  std::ifstream in(argv[1]);
  std::FILE * out = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "w");
  if (out) std::fprintf(out, "t,x,y,yaw,v,w,gyro_bias,speed_scale,yaw_bias,vy,ay,ay_bias,jump\n");
  std::FILE * innov = innov_path.empty() ? nullptr : std::fopen(innov_path.c_str(), "w");
  if (innov) std::fprintf(innov, "t,recv,along,cross,yaw,v,w,rejected\n");

  DelayedEkf ekf(p, k["history"]);
  VoVelocity vo(k["vo_window"], k["vo_period"]);
  int n_vo = 0, n_vo_rej = 0;
  double vesc_speed = 0.0, vesc_stamp = -1.0, next_out = 0.0, t_first = -1.0;
  int n_pose = 0, n_rej = 0, consecutive = 0, resets = 0, n_eval = 0;
  double sum_e2 = 0, sum_ey2 = 0, max_e = 0, max_jump = 0, sum_along2 = 0, sum_cross2 = 0;
  bool have_last = false;
  StateVec last_out;
  double last_out_t = 0;

  std::string line;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string type, f;
    std::getline(ss, type, ',');
    std::vector<double> v;
    while (std::getline(ss, f, ',')) v.push_back(std::stod(f));
    const double recv = v[0], stamp = v[1];
    if (t_first < 0) t_first = recv;

    // 50 Hz output, as the node's timer: newest state predicted to "now".
    while (ekf.initialized() && next_out <= recv) {
      StateVec x;
      StateMat P;
      ekf.predict_to(next_out, 0.1, x, P);
      if (have_last) {
        const double dt = next_out - last_out_t;
        // Jump = displacement not explained by the velocity vector (forward +
        // lateral, mean of both outputs, in the world frame).
        auto vel = [&](const StateVec & z) {
          const double th = z(YAW) + z(YAW_BIAS);
          const double vl = p.lateral_dynamics ? z(VY) : p.lever_arm * z(WZ);
          return Eigen::Vector2d(z(VX) * std::cos(th) - vl * std::sin(th), z(VX) * std::sin(th) + vl * std::cos(th));
        };
        const Eigen::Vector2d dp(x(X) - last_out(X), x(Y) - last_out(Y));
        const double jump = (dp - 0.5 * (vel(x) + vel(last_out)) * dt).norm();
        if (next_out - t_first > k["warmup"]) max_jump = std::max(max_jump, jump);
        if (out) {
          std::fprintf(out, "%.4f,%.4f,%.4f,%.5f,%.3f,%.4f,%.5f,%.4f,%.4f,%.4f,%.3f,%.3f,%.4f\n", next_out, x(X), x(Y),
                       x(YAW), x(VX), x(WZ), x(GYRO_BIAS), x(SPEED_SCALE), x(YAW_BIAS), x(VY), x(AY), x(AY_BIAS), jump);
        }
      }
      last_out = x;
      last_out_t = next_out;
      have_last = true;
      next_out += 1.0 / k["output_rate"];
    }

    if (type == "vesc") {
      vesc_speed = std::abs(v[2]) < k["standstill_erpm"] ? 0.0 : v[2] / k["erpm_per_mps"];
      vesc_stamp = stamp;
    } else if (type == "imu") {
      if (!ekf.initialized() || vesc_stamp < 0 || std::abs(stamp - vesc_stamp) > 0.1) continue;
      Measurement m;
      m.t = stamp;
      m.type = Measurement::Type::TWIST;
      m.twist.z << vesc_speed, v[2];
      m.twist.R = Eigen::Vector2d(k["speed_std"] * k["speed_std"], k["yaw_rate_std"] * k["yaw_rate_std"]).asDiagonal();
      ekf.add(m);
      if (p.lateral_dynamics && v.size() > 3) {
        Measurement a;
        a.t = stamp;
        a.type = Measurement::Type::ACCEL;
        a.accel.z = v[3] / k["accel_scale"];
        a.accel.R = k["accel_std"] * k["accel_std"];
        ekf.add(a);
      }
    } else if (type == "vo" && k["vo"] > 0.5) {
      VoVelocity::Out o;
      if (!vo.add(stamp, v[2], v[3], v[4], o) || !ekf.initialized()) continue;
      Measurement m;
      m.t = o.t;
      m.type = Measurement::Type::VO;
      m.vo.z << o.v_fwd, o.v_left;
      m.vo.R = k["vo_std"] * k["vo_std"] * Eigen::Matrix2d::Identity();
      ++n_vo;
      if (ekf.add(m) == DelayedEkf::Result::REJECTED) ++n_vo_rej;
    } else if (type == "pose") {
      const double t = stamp - k["pose_stamp_offset"];
      if (!ekf.initialized()) {
        StateVec x0 = StateVec::Zero();
        x0 << v[2], v[3], v[4], k["init_yaw_bias"], 0, 0, k["init_gyro_bias"], k["init_speed_scale"], 0, 0, k["init_ay_bias"],
          k["init_vo_scale"];
        StateVec sd;
        sd << 0.1, 0.1, 0.05, 0.03, 1.0, 0.5, 0.01, 0.05, 0.2, 2.0, 0.5, 0.05;
        ekf.initialize(t, x0, sd.array().square().matrix().asDiagonal());
        next_out = recv;
        continue;
      }
      // Innovation: filter estimate at the pose time, before the update.
      StateVec xp;
      StateMat Pp;
      ekf.state_at(t, xp, Pp);
      // Split into along-track / cross-track (timing errors show along-track).
      const double th = xp(YAW) + xp(YAW_BIAS);
      // the position is compared with the state pose_position_delay earlier
      const double vl = p.lateral_dynamics ? xp(VY) : p.lever_arm * xp(WZ), dpos = k["pose_position_delay"];
      const double ex = v[2] - (xp(X) - dpos * (xp(VX) * std::cos(th) - vl * std::sin(th)));
      const double ey = v[3] - (xp(Y) - dpos * (xp(VX) * std::sin(th) + vl * std::cos(th)));
      const double e = std::hypot(ex, ey);
      const double eyaw = normalize_angle(v[4] - xp(YAW));
      const double along = ex * std::cos(th) + ey * std::sin(th);
      const double cross = -ex * std::sin(th) + ey * std::cos(th);

      Measurement m;
      m.t = t;
      m.type = Measurement::Type::POSE;
      m.pose.z << v[2], v[3], v[4];
      const double vxy = std::max(std::max(v[5], v[6]), k["pose_min_xy"] * k["pose_min_xy"]);
      const double vyaw = std::max(v[7], k["pose_min_yaw"] * k["pose_min_yaw"]);
      m.pose.R = Eigen::Vector3d(vxy, vxy, vyaw).asDiagonal();
      m.pose.time_stddev = k["pose_time_std"];
      m.pose.position_delay = k["pose_position_delay"];
      const auto r = ekf.add(m);
      ++n_pose;
      if (innov) {
        std::fprintf(innov, "%.4f,%.4f,%.4f,%.4f,%.5f,%.3f,%.4f,%d\n", t, recv, along, cross, eyaw, xp(VX), xp(WZ),
                     r == DelayedEkf::Result::REJECTED ? 1 : 0);
      }
      if (r == DelayedEkf::Result::REJECTED) {
        ++n_rej;
        if (++consecutive >= k["max_rejections"]) {
          StateVec x0 = ekf.state();
          x0(X) = v[2]; x0(Y) = v[3]; x0(YAW) = v[4];
          StateVec sd;
          sd << 0.1, 0.1, 0.05, 0.03, 1.0, 0.5, 0.01, 0.05, 0.2, 2.0, 0.5, 0.05;
          ekf.initialize(t, x0, sd.array().square().matrix().asDiagonal());
          consecutive = 0;
          ++resets;
        }
      } else {
        consecutive = 0;
      }
      if (recv - t_first > k["warmup"]) {
        sum_e2 += e * e;
        sum_ey2 += eyaw * eyaw;
        sum_along2 += along * along;
        sum_cross2 += cross * cross;
        max_e = std::max(max_e, e);
        ++n_eval;
      }
    }
  }
  if (out) std::fclose(out);
  if (innov) std::fclose(innov);

  const StateVec & x = ekf.state();
  std::printf("poses %d rejected %d resets %d | innov rms %.1f cm (along %.1f cross %.1f) max %.1f cm "
              "yaw %.2f deg | max jump %.1f cm | gyro_bias %.4f speed_scale %.3f yaw_bias %.3f\n",
              n_pose, n_rej, resets, 100 * std::sqrt(sum_e2 / n_eval), 100 * std::sqrt(sum_along2 / n_eval),
              100 * std::sqrt(sum_cross2 / n_eval), 100 * max_e, 180 / M_PI * std::sqrt(sum_ey2 / n_eval),
              100 * max_jump, x(GYRO_BIAS), x(SPEED_SCALE), x(YAW_BIAS));
  if (k["vo"] > 0.5) std::printf("vo %d rejected %d vo_scale %.3f\n", n_vo, n_vo_rej, x(VO_SCALE));
  return 0;
}
