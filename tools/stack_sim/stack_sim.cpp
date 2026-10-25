// stack_sim: the whole driving stack in closed loop WITHOUT ROS, against ground truth.
//
//   plant (the identified car) -> sensors (IMU, VESC, LiDAR pose as the particle filter
//   delivers it) -> twist_estimator + ekf_localizer (the same EKF core and node logic)
//   -> MPC (the same controller core, as mpc_node in localization:=ekf or :=pf)
//
// Unlike Gazebo it can switch on the errors the real car has and the EKF is meant to
// absorb: gyro bias drift, servo trim, wheel slip, understeer, PF glitches / a lost PF,
// and the old (now()-stamped) or the new scan-synchronous PF timing. Deterministic per
// seed and ~100x faster than real time, so a change is judged on many seeds.
//
//   stack_sim path.csv [key=value ...] [--series out.csv]
//
// Keys (default):
//   ctrl=ekf|pf (ekf)          what the MPC uses: the EKF output or the PF pose directly
//   pf=old|scan (old)          PF timing: old = stamped at publication, ~0.2 s late, 8.4 Hz;
//                              scan = stamped with the scan time, 9.7 Hz, ~95 ms late
//   speed=1.5 laps=3 seed=1 t_end=600
//   gyro_bias=0.0 gyro_bias_rw=0.0     rad/s initial bias, rad/s/sqrt(s) random walk
//   trim=0.0                   rad steering trim the controller does not know
//   kus=0.0                    understeer: yaw rate = v tan(d) / (L (1 + kus v^2))
//   wheel_slip=0.0             s: wheel speed = v + wheel_slip * dv/dt
//   dist=1                     yaw disturbance of the car (car_like.json), 0 = off
//   pf_noise=1                 scale of the PF pose errors (0 = perfect pose)
//   plant=kin|dyn (kin)        kin: the identified kinematic car (never slides); dyn: single-track
//                              model with Fiala tyres, friction circle at the driven rear axle
//   mu=1.0 mu_wet=0.4          friction coefficient; mu_wet on the path stretch wet_s0..wet_s1 [m]
//   mass=4.0 iz=0.055          kg, kg m^2 (assumed: F1TENTH-like, not identified on this car)
//   pf_pos_extra=0.04          s the PF position is older than its heading (car_like.json; the
//                              2026-10-05 bags fit 0 better, see ekf_replay pose_position_delay)
//   glitch_rate=0 glitch_m=0.5 per-pose probability and size of a PF position jump
//   lost_t=-1 lost_dur=3 lost_rate=0.5  PF loses track at lost_t: pose slides back at lost_rate m/s
//   + any EKF / MPC parameter listed in apply_key() below
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "mpc_tracking_controller/tracking_mpc.hpp"
#include "rc_localization/ekf_core.hpp"

namespace rl = rc_localization;
using mpc_tc::ControllerParams;
using mpc_tc::PathGeometry;
using mpc_tc::Projection;
using mpc_tc::TrackingController;

static bool read_path(const std::string& fn, std::vector<double>& x, std::vector<double>& y) {
  std::ifstream f(fn);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || !(std::isdigit(line[0]) || line[0] == '-' || line[0] == '.')) continue;
    std::stringstream ss(line);
    std::string a, b;
    std::getline(ss, a, ',');
    std::getline(ss, b, ',');
    x.push_back(std::stod(a));
    y.push_back(std::stod(b));
  }
  return x.size() > 3;
}

struct Delayed {  // dead time + first-order lag (as closed_loop.cpp)
  Delayed(double d, double ta, double g) : dead(d), tau(ta), gain(g) {}
  double dead, tau, gain, x = 0.0, u = 0.0;
  std::deque<std::pair<double, double>> q;
  void cmd(double t, double c) { q.emplace_back(t + dead, c); }
  void step(double t, double h) {
    while (!q.empty() && q.front().first <= t) {
      u = q.front().second;
      q.pop_front();
    }
    x = gain * u + (x - gain * u) * std::exp(-h / tau);
  }
};

struct Ou {  // Ornstein-Uhlenbeck process, exact discretisation
  double sigma = 0.0, tau = 1.0, x = 0.0;
  void step(double dt, std::mt19937& g) {
    std::normal_distribution<double> n(0.0, 1.0);
    const double a = std::exp(-dt / tau);
    x = a * x + sigma * std::sqrt(1.0 - a * a) * n(g);
  }
};

struct Event {
  double arr;  // arrival time at the receiving node
  int seq;     // FIFO among equal arrival times
  std::function<void()> fn;
  bool operator>(const Event& o) const { return arr > o.arr || (arr == o.arr && seq > o.seq); }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s path.csv [key=value ...] [--series out.csv]\n", argv[0]);
    return 2;
  }
  std::vector<double> px, py;
  if (!read_path(argv[1], px, py)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 2;
  }
  std::map<std::string, std::string> kv;
  std::string series;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--series") && i + 1 < argc) {
      series = argv[++i];
      continue;
    }
    const char* eq = std::strchr(argv[i], '=');
    if (!eq) {
      std::fprintf(stderr, "bad argument %s\n", argv[i]);
      return 2;
    }
    kv[std::string(argv[i], static_cast<size_t>(eq - argv[i]))] = eq + 1;
  }
  auto num = [&](const char* k, double d) {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    const double v = std::stod(it->second);
    kv.erase(it);
    return v;
  };
  auto str = [&](const char* k, const char* d) {
    auto it = kv.find(k);
    if (it == kv.end()) return std::string(d);
    const std::string v = it->second;
    kv.erase(it);
    return v;
  };

  // ------------------------------------------------------------------ scenario
  const std::string ctrl_mode = str("ctrl", "ekf"), pf_mode = str("pf", "old");
  const bool scan_sync = pf_mode == "scan";
  if ((ctrl_mode != "ekf" && ctrl_mode != "pf") || (pf_mode != "old" && pf_mode != "scan")) {
    std::fprintf(stderr, "ctrl=ekf|pf, pf=old|scan\n");
    return 2;
  }
  const double speed = num("speed", 1.5), laps = num("laps", 3), t_end = num("t_end", 600);
  const unsigned seed = static_cast<unsigned>(num("seed", 1));
  const double gyro_bias0 = num("gyro_bias", 0.0), gyro_bias_rw = num("gyro_bias_rw", 0.0);
  const double trim = num("trim", 0.0), kus = num("kus", 0.0), wheel_slip = num("wheel_slip", 0.0);
  const bool dist_on = num("dist", 1) != 0.0;
  // dynamic plant: single-track model with Fiala tyres and a friction coefficient
  const std::string plant = str("plant", "kin");
  if (plant != "kin" && plant != "dyn") {
    std::fprintf(stderr, "plant=kin|dyn\n");
    return 2;
  }
  const bool dyn = plant == "dyn";
  const double mu_dry = num("mu", 1.0), mu_wet = num("mu_wet", 0.4);
  const double wet_s0 = num("wet_s0", -1.0), wet_s1 = num("wet_s1", -1.0);  // wet stretch of the path [m]
  const double car_mass = num("mass", 4.0), car_iz = num("iz", 0.055);
  double tyre_cs = 1.0;  // scale of the tyre cornering stiffness (assumed values: test both ways)
  const double pf_noise = num("pf_noise", 1.0);  // scale of every PF pose error (0 = perfect pose)
  const double pf_pos_extra = num("pf_pos_extra", 0.04);  // s position older than heading (car_like.json)
  const double glitch_rate = num("glitch_rate", 0.0), glitch_m = num("glitch_m", 0.5);
  const double lost_t = num("lost_t", -1.0), lost_dur = num("lost_dur", 3.0), lost_rate = num("lost_rate", 0.5);

  // the car (RC_Car_Sim car_like.json, identified on fair_bags)
  const double L = 0.28, A = 0.17, heading_bias = 0.045, pose_per_vesc = 1.04, erpm_per_mps_true = 2171.0;
  const double gyro_std = 0.049, accel_std = 1.0;
  Ou dist_ou{0.0116 * 1.23, 0.032};  // steering disturbance (sigma x Gazebo gain: no Gazebo here)
  double dist_lp = 0.0;
  // PF pose errors (car_like.json pose): lateral OU + white, longitudinal OU, heading white
  Ou lat_ou{0.039 * pf_noise, 5.0}, lon_ou{0.065 * pf_noise, 0.21};
  const double lat_white = 0.0127 * pf_noise, hdg_white = 0.0047 * pf_noise, pos_extra_latency = pf_pos_extra;
  // timing
  const double T_pose = scan_sync ? 1.0 / 9.68 : 1.0 / 8.44, pose_period_jitter = scan_sync ? 0.002 : 0.0135;
  const double pose_latency = scan_sync ? 0.095 : 0.200, pose_latency_jitter = scan_sync ? 0.010 : 0.0127;
  const double scan_stamp_error = 0.025;  // scan-sync PF: the pose is ~25 ms older than its stamp (replay)
  const double T_imu = 1.0 / 197.0, imu_latency = 0.005, T_vesc = 0.02, vesc_age = 0.01, t_compute = 0.014;

  // ------------------------------------------------------------------ EKF (ekf_localizer.yaml values)
  rl::EkfParams ep;
  ep.lever_arm = 0.17;
  double pose_stamp_offset = scan_sync ? scan_stamp_error : 0.20, pose_time_stddev = scan_sync ? 0.01 : 0.04;
  double pose_position_delay = 0.0;
  double pose_min_xy = 0.03, pose_min_yaw = 0.015, twist_min_vx = 0.05, twist_min_wz = 0.01;
  double erpm_per_mps = 2080.0, speed_stddev = 0.1, yaw_rate_stddev = 0.02, standstill_erpm = 50.0;
  double init_yaw_bias = 0.05, max_extrapolation = 0.1, history = 1.0, output_rate = 50.0;
  int max_rejections = 15;
  // ------------------------------------------------------------------ MPC (mpc_car.yaml values)
  ControllerParams cp;
  cp.mpc.v_max = speed;
  double speed_cmd_per_mps = 2171.0, mpc_horizon = cp.mpc.horizon, grip_on = 0.0;

  // generic numeric overrides: key=value for the parameters below
  std::map<std::string, double*> keys = {
      {"ekf.q_pos", &ep.q_pos}, {"ekf.q_yaw", &ep.q_yaw}, {"ekf.q_yaw_bias", &ep.q_yaw_bias},
      {"ekf.q_vx", &ep.q_vx}, {"ekf.q_wz", &ep.q_wz}, {"ekf.q_gyro_bias", &ep.q_gyro_bias},
      {"ekf.q_speed_scale", &ep.q_speed_scale}, {"ekf.pose_gate", &ep.pose_gate},
      {"ekf.pose_stamp_offset", &pose_stamp_offset}, {"ekf.pose_time_stddev", &pose_time_stddev}, {"ekf.pose_position_delay", &pose_position_delay},
      {"ekf.pose_min_xy", &pose_min_xy}, {"ekf.pose_min_yaw", &pose_min_yaw}, {"ekf.erpm_per_mps", &erpm_per_mps},
      {"ekf.init_yaw_bias", &init_yaw_bias}, {"ekf.max_extrapolation", &max_extrapolation},
      {"ekf.output_rate", &output_rate},
      {"mpc.v_max", &cp.mpc.v_max}, {"mpc.a_lat_max", &cp.mpc.a_lat_max}, {"mpc.max_e_lat", &cp.mpc.max_e_lat},
      {"mpc.max_e_psi", &cp.mpc.max_e_psi}, {"mpc.control_period", &cp.control_period},
      {"mpc.heading_bias", &cp.heading_bias}, {"mpc.dt", &cp.mpc.dt}, {"mpc.steer_offset_tau", &cp.steer_offset_tau},
      {"mpc.grip", &grip_on}, {"mpc.grip_margin", &cp.grip_margin}, {"mpc.grip_recover_tau", &cp.grip_recover_tau},
      {"mpc.slip_rel", &cp.slip_rel}, {"mpc.slip_abs", &cp.slip_abs}, {"mpc.grip_probe_tau", &cp.grip_probe_tau}, {"mpc.grip_probe_level", &cp.grip_probe_level},
      {"mpc.slip_min_time", &cp.slip_min_time}, {"mpc.understeer", &cp.model.understeer}, {"mpc.understeer_tau", &cp.understeer_tau},
      {"tyre_cs", &tyre_cs}, {"mpc.horizon", &mpc_horizon}, {"mpc.speed_cmd_per_mps", &speed_cmd_per_mps},
  };
  for (auto it = kv.begin(); it != kv.end();) {
    auto k = keys.find(it->first);
    if (k == keys.end()) {
      std::fprintf(stderr, "unknown key %s\n", it->first.c_str());
      return 2;
    }
    *k->second = std::stod(it->second);
    it = kv.erase(it);
  }

  cp.mpc.horizon = static_cast<int>(mpc_horizon);
  cp.grip_adaptation = grip_on != 0.0;
  // MPC exactly as mpc_node + mpc.launch.py set it up
  if (ctrl_mode == "ekf") {
    cp.pose_latency_position = 0.0;
    cp.pose_latency_heading = 0.0;
  } else if (scan_sync) {
    cp.pose_latency_position = cp.pose_latency_heading = 0.025;
  }
  const bool pose_latency_from_stamp = ctrl_mode == "pf" && scan_sync;
  {
    const TrackingController tmp(cp);
    cp.mpc.delta_lo = tmp.steer_to_eff(-cp.delta_cmd_max);
    cp.mpc.delta_hi = tmp.steer_to_eff(cp.delta_cmd_max);
  }
  cp.mpc.speed_delay_steps = static_cast<int>(std::lround((cp.speed_dead_time - cp.steer_dead_time) / cp.mpc.dt));
  TrackingController ctrl(cp);
  ctrl.set_path(px, py);
  PathGeometry truth(px, py);

  rl::DelayedEkf ekf(ep, history);

  // one stream per noise source: a scenario that adds a source leaves the others unchanged
  std::mt19937 rng(seed), rng_imu(seed + 1000), rng_pose(seed + 2000), rng_glitch(seed + 3000), rng_bias(seed + 4000);
  std::normal_distribution<double> N01(0.0, 1.0);
  std::uniform_real_distribution<double> U01(0.0, 1.0);

  // ------------------------------------------------------------------ plant state (rear axle)
  double x, y, psi;
  truth.point_at(5.0, x, y);
  psi = truth.psi_at(5.0);
  x -= A * std::cos(psi);
  y -= A * std::sin(psi);
  Delayed steer(0.05, 0.04, 1.0), spd(0.18, 0.25, 1.005);
  struct Hist { double t, x, y, psi, v, vw; };
  std::deque<Hist> hist;
  auto at = [&](double tq) {
    for (auto it = hist.rbegin(); it != hist.rend(); ++it)
      if (it->t <= tq) return *it;
    return hist.front();
  };
  double gyro_bias = gyro_bias0, v_prev = 0.0;
  // dynamic plant: body velocities at the centre of gravity. Tyre data: F1TENTH single-track model
  // (C_alpha = C_S mu F_z, C_Sf 4.718, C_Sr 5.456 1/rad), scaled to this wheelbase. Mildly
  // understeering (~0.03 rad/g), i.e. kinematic at the bags' 0.1-0.3 g, as identified.
  const double lr = 0.145, lf = L - lr, grav = 9.81;
  const double Fzf = car_mass * grav * lr / L, Fzr = car_mass * grav * lf / L;
  double dvx = 0.0, dvy = 0.0, dr = 0.0, mu_now = mu_dry, ay_true = 0.0, beta_max = 0.0, slide_now = 0.0;
  auto fiala = [](double alpha, double c_alpha, double fmax) {
    // brush tyre: linear for small slip, saturating at fmax (= mu F_z or what the friction
    // circle leaves) beyond alpha_sl
    if (fmax <= 1e-9) return 0.0;
    const double z = std::tan(alpha), zsl = 3.0 * fmax / c_alpha;
    if (std::fabs(z) >= zsl) return std::copysign(fmax, alpha);
    return c_alpha * z - c_alpha * c_alpha / (3.0 * fmax) * std::fabs(z) * z +
           c_alpha * c_alpha * c_alpha / (27.0 * fmax * fmax) * z * z * z;
  };

  // ------------------------------------------------------------------ event queue
  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;
  int seq = 0;
  double now_evt = 0.0;  // arrival time of the event being processed
  auto post = [&](double arr, std::function<void()> fn) { events.push({arr, seq++, std::move(fn)}); };

  // twist_estimator state
  double te_speed = 0.0, te_vesc_stamp = -1.0;
  // ekf_localizer state
  int consecutive_rej = 0, total_rej = 0, resets = 0;
  auto ekf_init = [&](double t, double mx, double my, double myaw) {
    rl::StateVec x0 = rl::StateVec::Zero();
    x0(rl::X) = mx;
    x0(rl::Y) = my;
    x0(rl::YAW) = myaw;
    x0(rl::YAW_BIAS) = init_yaw_bias;
    x0(rl::SPEED_SCALE) = 1.0;
    if (ekf.initialized()) {
      for (int i : {rl::YAW_BIAS, rl::GYRO_BIAS, rl::SPEED_SCALE, rl::AY_BIAS, rl::VX, rl::WZ}) x0(i) = ekf.state()(i);
    }
    rl::StateVec sd;
    sd << 0.1, 0.1, 0.05, 0.03, 1.0, 0.5, 0.01, 0.05, 0.2, 2.0, 0.5;
    ekf.initialize(t, x0, sd.array().square().matrix().asDiagonal());
    consecutive_rej = 0;
  };

  // metrics
  std::vector<double> el, el_st, el_cu, d_st, ekf_pos_err, ekf_hdg_err, jumps;
  double last_out_x = NAN, last_out_y = NAN, last_out_t = 0.0;
  int aborts = 0;
  double last_dcmd = 0.0, s_trav = 0.0;
  std::ofstream ser;
  if (!series.empty()) {
    ser.open(series);
    ser << "t,s,e_lat,v,delta_cmd,ekf_err,ekf_hdg_err,gyro_bias_est,yaw_bias_est,slide,grip,mu,ay\n";
  }

  // controller step on a pose (pose = PF pose or EKF output), as mpc_node::on_odom
  double last_step = -1.0, pend_v = 0.0, pend_d = 0.0;
  bool stopped = false;
  auto control = [&](double t_arr, double stamp, double mx, double my, double myaw) {
    if (stopped) return;
    if (last_step > 0.0 && t_arr - last_step < 0.9 * cp.control_period) return;
    last_step = t_arr;
    const double pose_age = pose_latency_from_stamp ? std::clamp(t_arr - stamp, 0.0, 0.5) : 0.0;
    const auto o = ctrl.on_pose(t_arr, mx, my, myaw, speed, pose_age);
    if (o.status != mpc_tc::Status::kOk) {
      if (o.status == mpc_tc::Status::kAbort) {
        ++aborts;
        stopped = true;  // the node disarms
      }
      pend_v = pend_d = 0.0;
    } else {
      pend_v = o.v_cmd;
      pend_d = o.delta_cmd;
    }
    const double v_c = pend_v, d_c = pend_d;
    post(t_arr + t_compute, [&, v_c, d_c] {
      // publication: the controller logs it, the car receives it
      const double tp = now_evt;
      ctrl.on_command_sent(tp, v_c, d_c);
      steer.cmd(tp, d_c);
      spd.cmd(tp, v_c);
      last_dcmd = d_c;
    });
  };

  const double h = 0.001;
  double t = 0.0, next_imu = 0.0, next_vesc = 0.0, next_pose = 0.5, next_out = 0.5, next_metric = 0.0;
  double last_s = truth.project(x + A * std::cos(psi), y + A * std::sin(psi), psi).s;
  double lost_offset = 0.0;
  while (t < t_end && s_trav < laps * truth.length() && !stopped) {
    // ---------------- plant
    steer.step(t, h);
    spd.step(t, h);
    if (dist_on) {
      dist_ou.step(h, rng);
      dist_lp += (dist_ou.x - dist_lp) * (1.0 - std::exp(-h / 0.032));
    }
    const double delta = ctrl.steer_to_eff(steer.x) + trim + (dist_on ? dist_lp : 0.0);
    double v, r, v_wheel;
    if (!dyn) {
      v = spd.x;
      const double accel = (v - v_prev) / h;
      v_prev = v;
      r = v * std::tan(delta) / (L * (1.0 + kus * v * v));
      x += h * v * std::cos(psi + 0.5 * h * r);
      y += h * v * std::sin(psi + 0.5 * h * r);
      psi += h * r;
      v_wheel = v + wheel_slip * accel;
      ay_true = v * r;
    } else {
      // the VESC regulates the wheel speed (the identified speed dynamics); the car follows through
      // the rear tyre's longitudinal force; lateral forces from the slip angles
      v_wheel = spd.x;
      const double vxs = std::max(dvx, 0.5);
      const double kappa = (v_wheel - dvx) / vxs;
      const double fr_max = mu_now * Fzr, ff_max = mu_now * Fzf;
      const double fx = std::clamp(20.0 * Fzr * kappa, -fr_max, fr_max);
      const double fyr_max = std::sqrt(std::max(0.0, fr_max * fr_max - fx * fx));
      const double af = delta - std::atan2(dvy + lf * dr, vxs), ar = -std::atan2(dvy - lr * dr, vxs);
      const double fyf = fiala(af, tyre_cs * 4.718 * mu_now * Fzf, ff_max), fyr = fiala(ar, tyre_cs * 5.456 * mu_now * Fzr, fyr_max);
      const double ax = (fx - fyf * std::sin(delta)) / car_mass + dvy * dr;
      const double vyd = (fyr + fyf * std::cos(delta)) / car_mass - dvx * dr;
      const double rd = (lf * fyf * std::cos(delta) - lr * fyr) / car_iz;
      dvx = std::max(0.0, dvx + h * ax);
      dvy += h * vyd;
      dr += h * rd;
      // below ~0.6 m/s the slip angles are ill-defined: blend into the kinematic model
      const double w = std::clamp((dvx - 0.3) / 0.3, 0.0, 1.0);
      const double r_kin = dvx * std::tan(delta) / L;
      dr = w * dr + (1.0 - w) * r_kin;
      dvy = w * dvy + (1.0 - w) * lr * r_kin;
      ay_true = vyd + dvx * dr;
      // move the centre of gravity, keep (x, y) = rear axle as in the kinematic plant
      double cx = x + lr * std::cos(psi), cy = y + lr * std::sin(psi);
      cx += h * (dvx * std::cos(psi) - dvy * std::sin(psi));
      cy += h * (dvx * std::sin(psi) + dvy * std::cos(psi));
      psi += h * dr;
      x = cx - lr * std::cos(psi);
      y = cy - lr * std::sin(psi);
      v = dvx;
      r = dr;
      // slide: body slip angle beyond the kinematic one (lr tan(delta) / L at the CG)
      slide_now = std::atan2(dvy, std::max(dvx, 0.5)) - std::atan(lr * std::tan(delta) / L);
      if (t > 10.0)
        beta_max = std::max(beta_max, std::fabs(std::atan2(dvy, std::max(dvx, 0.5)) - std::atan(lr * std::tan(delta) / L)));
    }
    t += h;
    gyro_bias += gyro_bias_rw * std::sqrt(h) * N01(rng_bias);
    hist.push_back({t, x, y, psi, v, v_wheel});
    while (hist.size() > 2 && hist.front().t < t - 1.5) hist.pop_front();

    // ---------------- sensors
    if (t >= next_imu) {
      next_imu += T_imu;
      const double wz = r + gyro_bias + gyro_std * N01(rng_imu), ay = ay_true + accel_std * N01(rng_imu);
      const double stamp = t;
      (void)ay;
      ctrl.on_imu(t + imu_latency, wz);  // mpc_node stamps the IMU with now() on reception
      post(t + imu_latency, [&, stamp, wz] {
        // twist_estimator: one twist per IMU message with the latest VESC speed
        if (te_vesc_stamp < 0.0 || std::fabs(stamp - te_vesc_stamp) > 0.1) return;
        const double tw_v = te_speed;
        post(now_evt + 0.001, [&, stamp, tw_v, wz] {
          if (!ekf.initialized()) return;
          rl::Measurement m;
          m.t = stamp;
          m.type = rl::Measurement::Type::TWIST;
          m.twist.z << tw_v, wz;
          m.twist.R = Eigen::Vector2d(std::max(speed_stddev * speed_stddev, twist_min_vx * twist_min_vx),
                                      std::max(yaw_rate_stddev * yaw_rate_stddev, twist_min_wz * twist_min_wz))
                          .asDiagonal();
          ekf.add(m);
        });
      });
    }
    if (t >= next_vesc) {
      next_vesc += T_vesc;
      const double erpm = at(t - vesc_age).vw / pose_per_vesc * erpm_per_mps_true;
      post(t, [&, erpm] {
        const double tr = now_evt;  // vesc_driver stamps on reception
        te_speed = std::fabs(erpm) < standstill_erpm ? 0.0 : erpm / erpm_per_mps;
        te_vesc_stamp = tr;
        ctrl.on_speed(tr, erpm / speed_cmd_per_mps);
      });
    }
    if (t >= next_pose) {
      const double dtp = T_pose + pose_period_jitter * N01(rng_pose);
      next_pose += std::max(0.05, dtp);
      lat_ou.step(dtp, rng_pose);
      lon_ou.step(dtp, rng_pose);
      // the scan is taken now; the PF pose describes t_h (heading) / t_h - extra (position)
      const double t_h = scan_sync ? t - scan_stamp_error : t;
      const Hist hh = at(t_h), hp = at(t_h - pos_extra_latency);
      const double c = std::cos(hp.psi), s = std::sin(hp.psi);
      const double nl = lat_ou.x + lat_white * N01(rng_pose), nn = lon_ou.x;
      double mx = hp.x + A * c - s * nl + c * nn, my = hp.y + A * s + c * nl + s * nn;
      const double myaw = hh.psi - heading_bias + hdg_white * N01(rng_pose);
      if (glitch_rate > 0.0 && U01(rng_glitch) < glitch_rate) {
        const double ang = 2.0 * M_PI * U01(rng_glitch);
        mx += glitch_m * std::cos(ang);
        my += glitch_m * std::sin(ang);
      }
      if (lost_t >= 0.0 && t >= lost_t && t < lost_t + lost_dur) lost_offset += lost_rate * dtp;
      else lost_offset = 0.0;
      mx -= c * lost_offset;
      my -= s * lost_offset;
      const double arr = t + pose_latency + pose_latency_jitter * N01(rng_pose);
      const double stamp = scan_sync ? t : arr;
      post(arr, [&, stamp, mx, my, myaw] {
        const double ta = now_evt;
        // ekf_localizer::on_pose
        const double tm = stamp - pose_stamp_offset;
        if (!ekf.initialized()) {
          ekf_init(tm, mx, my, myaw);
        } else {
          rl::Measurement m;
          m.t = tm;
          m.type = rl::Measurement::Type::POSE;
          m.pose.z << mx, my, myaw;
          m.pose.R = Eigen::Vector3d(pose_min_xy * pose_min_xy, pose_min_xy * pose_min_xy, pose_min_yaw * pose_min_yaw)
                         .asDiagonal();
          m.pose.time_stddev = pose_time_stddev;
          m.pose.position_delay = pose_position_delay;
          const auto res = ekf.add(m);
          if (res == rl::DelayedEkf::Result::APPLIED) {
            consecutive_rej = 0;
          } else if (res == rl::DelayedEkf::Result::REJECTED) {
            ++total_rej;
            if (++consecutive_rej >= max_rejections) {
              ekf_init(tm, mx, my, myaw);
              ++resets;
            }
          }
        }
        if (ctrl_mode == "pf") control(ta, stamp, mx, my, myaw);
      });
    }

    // ---------------- process everything that has arrived
    while (!events.empty() && events.top().arr <= t) {
      const Event e = events.top();
      events.pop();
      now_evt = e.arr;
      e.fn();
    }

    // ---------------- EKF output timer (ekf_localizer::publish_output)
    if (t >= next_out) {
      next_out += 1.0 / output_rate;
      if (ekf.initialized()) {
        rl::StateVec xe;
        rl::StateMat Pe;
        ekf.predict_to(t, max_extrapolation, xe, Pe);
        if (!xe.allFinite() || Pe(rl::X, rl::X) + Pe(rl::Y, rl::Y) > 4.0) {
          ekf.reset();
          ++resets;
        } else {
          // error of the published pose: base_link and LiDAR heading now
          const double bx = x + A * std::cos(psi), by = y + A * std::sin(psi);
          const double ex = xe(rl::X) - bx, ey = xe(rl::Y) - by;
          if (t > 10.0) {
            ekf_pos_err.push_back(std::hypot(ex, ey));
            ekf_hdg_err.push_back(rl::normalize_angle(xe(rl::YAW) - (psi - heading_bias)));
            if (!std::isnan(last_out_x)) {
              // jump: change between outputs beyond what the true motion explains
              const double dt_o = t - last_out_t;
              jumps.push_back(std::hypot(xe(rl::X) - last_out_x - v * std::cos(psi) * dt_o,
                                         xe(rl::Y) - last_out_y - v * std::sin(psi) * dt_o));
            }
          }
          last_out_x = xe(rl::X);
          last_out_y = xe(rl::Y);
          last_out_t = t;
          if (ctrl_mode == "ekf") control(t, t, xe(rl::X), xe(rl::Y), xe(rl::YAW));
        }
      }
    }

    // ---------------- tracking metrics on the true state (50 Hz)
    if (t >= next_metric) {
      next_metric += 0.02;
      const Projection tp = truth.project(x + A * std::cos(psi), y + A * std::sin(psi), psi);
      {
        const double sm = std::fmod(tp.s, truth.length());
        mu_now = (wet_s0 >= 0.0 && sm >= wet_s0 && sm <= wet_s1) ? mu_wet : mu_dry;
      }
      double ds = tp.s - last_s;
      if (ds < -0.5 * truth.length()) ds += truth.length();
      if (ds > 0.5 * truth.length()) ds -= truth.length();
      s_trav += ds;
      last_s = tp.s;
      if (t > 10.0) {
        double kmax = 0.0;
        for (double q = -0.5; q <= 0.5; q += 0.1) kmax = std::max(kmax, std::fabs(truth.kappa_at(tp.s + q)));
        el.push_back(tp.e_lat);
        if (kmax < 0.15) {
          el_st.push_back(tp.e_lat);
          d_st.push_back(last_dcmd);
        }
        if (std::fabs(tp.kappa) > 0.6) el_cu.push_back(tp.e_lat);
      }
      if (ser) {
        const rl::StateVec xs = ekf.initialized() ? ekf.state() : rl::StateVec::Zero();
        ser << t << "," << tp.s << "," << tp.e_lat << "," << v << "," << last_dcmd << ","
            << (ekf_pos_err.empty() ? 0.0 : ekf_pos_err.back()) << "," << (ekf_hdg_err.empty() ? 0.0 : ekf_hdg_err.back())
            << "," << xs(rl::GYRO_BIAS) << "," << xs(rl::YAW_BIAS) << "," << slide_now << "," << ctrl.grip() << ","
            << mu_now << "," << ay_true << "\n";
      }
    }
  }

  auto rms = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double e : v) s += e * e;
    return v.empty() ? NAN : std::sqrt(s / v.size());
  };
  auto mean = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double e : v) s += e;
    return v.empty() ? NAN : s / v.size();
  };
  auto absmax = [](const std::vector<double>& v) {
    double m = 0.0;
    for (double e : v) m = std::max(m, std::fabs(e));
    return m;
  };
  auto stdv = [&](const std::vector<double>& v) {
    const double m = mean(v);
    double s = 0.0;
    for (double e : v) s += (e - m) * (e - m);
    return v.empty() ? NAN : std::sqrt(s / v.size());
  };
  // one machine-readable line: key=value
  std::printf("laps=%.2f time=%.1f elat_rms=%.4f elat_max=%.4f elat_st=%.4f elat_st_mean=%.4f elat_cu=%.4f "
              "steer_std_st=%.4f ekf_err=%.4f ekf_err_max=%.4f ekf_hdg=%.4f jump_max=%.4f rejected=%d resets=%d "
              "aborts=%d beta_max=%.4f\n",
              s_trav / truth.length(), t, rms(el), absmax(el), rms(el_st), mean(el_st), rms(el_cu), stdv(d_st),
              mean(ekf_pos_err), absmax(ekf_pos_err), rms(ekf_hdg_err), absmax(jumps), total_rej, resets, aborts, beta_max);
  return 0;
}
