// Camera velocity from visual odometry POSES (the ZED on the car publishes its
// odometry with an empty twist), without ROS. The velocity over a window of
// `window` seconds, expressed in the camera's frame at the window's middle and
// stamped there; one value every `period` seconds (overlapping windows give
// correlated noise, so not at the full 30 Hz).
//
// 2026-10-19 bags (ZED 30 Hz, window 0.2 s): scale 1.043-1.045 vs the wheels,
// lateral velocity 0.16-0.22 wz (kinematic: 0.17 wz), timing within 15 ms of the
// gyro, noise ~0.1 m/s lateral and ~0.15 m/s forward.

#pragma once

#include <cmath>
#include <deque>

namespace rc_localization
{

class VoVelocity
{
public:
  struct Out
  {
    double t, v_fwd, v_left;
  };

  VoVelocity(double window, double period) : window_(window), period_(period) {}

  // Adds a pose; returns true and fills `out` when a new velocity is due.
  bool add(double t, double x, double y, double yaw, Out & out)
  {
    if (!buf_.empty() && (t <= buf_.back().t || t - buf_.back().t > window_)) {
      buf_.clear();   // out of order or a gap: start again
    }
    if (!buf_.empty()) {
      // continuous yaw across +-pi
      yaw = buf_.back().yaw + std::remainder(yaw - buf_.back().yaw, 2.0 * M_PI);
    }
    buf_.push_back({t, x, y, yaw});
    while (buf_.size() > 2 && t - buf_[1].t >= window_) buf_.pop_front();
    const Pose & a = buf_.front();
    // 5 ms slack: poses spaced 1/30 s must not miss a 0.1 s period by rounding
    if (t - a.t < window_ - 0.005 || t < last_out_ + period_ - 0.005) return false;
    const double dt = t - a.t;
    const double vx = (x - a.x) / dt, vy = (y - a.y) / dt, ym = 0.5 * (yaw + a.yaw);
    out.t = 0.5 * (t + a.t);
    out.v_fwd = std::cos(ym) * vx + std::sin(ym) * vy;
    out.v_left = -std::sin(ym) * vx + std::cos(ym) * vy;
    last_out_ = t;
    return true;
  }

private:
  struct Pose
  {
    double t, x, y, yaw;
  };
  double window_, period_, last_out_ = -1e18;
  std::deque<Pose> buf_;
};

}  // namespace rc_localization
