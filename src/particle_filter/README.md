# Particle Filter Localization

This code implements the MCL algorithm for the RACECAR. 

[![YouTube Demo](./media/thumb.jpg)](https://www.youtube.com/watch?v=-c_0hSjgLYw)

For high efficiency in Python, it uses Numpy arrays and [RangeLibc](https://github.com/f1tenth/range_libc) for fast 2D ray casting.

# Installation

To run this, you need to ensure that both the map_server ROS package, and the python wrappers for RangeLibc are installed.

For the map server:
```
sudo apt-get update
rosdep install -r --from-paths src --ignore-src --rosdistro kinetic -y
```

For [RangeLibc](https://github.com/f1tenth/range_libc):

```
sudo pip install cython
git clone https://github.com/f1tenth/range_libc
cd range_libc/pywrapper
# on VM
./compile.sh
# on car - compiles GPU ray casting methods
./compile_with_cuda.sh
```

# Usage

The majority of parameters you might want to tweak are in the launch/localize.launch file. You may have to modify the "odometry_topic" or "scan_topic" parameters to match your environment.

```
roslaunch particle_filter localize.launch
```

Once the particle filter is running, you can visualize the map and other particle filter visualization message in RViz. Use the "2D Pose Estimate" tool from the RViz toolbar to initialize the particle locations.

See [launch/localize.launch](/particle_filter/launch/localize.launch) for docs on available parameters and arguments.

The "range_method" parameter determines which RangeLibc ray casting method to use. The default is cddt because it is fast and has a low initialization time. The fastest option on the CPU is "glt" but it has a slow startup. The fastest version if you have can compile RangeLibc with CUDA enabled is "rmgpu". See this performance comparison chart:

![Range Method Performance Comparison](./media/comparison.png)

# This fork: launch modes and offline replay

```
ros2 launch particle_filter localize_launch.py                  # original: updates driven by the odometry (10 Hz throttle), stamped now()
ros2 launch particle_filter localize_launch.py scan_sync:=true  # one update per scan, motion up to the scan's measurement time, pose stamped with it
ros2 launch particle_filter localize_launch.py ekf:=true        # pose source of rc_localization (scan_sync + twist motion model, no TF)
```
`motion_noise:=proportional` (any mode) adds the motion noise in the car frame, proportional to the motion
since the last update, instead of the fixed `motion_dispersion_*` every update (less jitter, see the commit).

With `scan_sync:=true` the consumers must use the stamp: `ekf_localization.launch.py pose_stamp_offset:=0.0`,
`mpc.launch.py localization:=pf pf_scan_sync:=true`.

[tools/](tools/README.md) runs this node over a recorded rosbag on a PC (CPU range_libc) to compare
parameters / code changes without the car: timing, scan reuse, pose lag and consistency metrics.

# Docs

This code is the staff solution to the lab guide found in the [/docs](/particle_filter/docs) folder. A mathematical derivation of MCL is available in that guide.

There is also documentation on RangeLibc in the [/docs](/particle_filter/docs) folder.

The code itself also contains comments describing purpose of each method.

# Cite

This library accompanies the following [publication](http://arxiv.org/abs/1705.01167).

    @article{walsh17,
        author = {Corey Walsh and 
                  Sertac Karaman},
        title  = {CDDT: Fast Approximate 2D Ray Casting for Accelerated Localization},
        volume = {abs/1705.01167},
        url    = {http://arxiv.org/abs/1705.01167},
        year   = {2017}}
