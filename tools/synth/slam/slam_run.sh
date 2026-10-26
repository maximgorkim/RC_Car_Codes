#!/bin/bash
# slam_run.sh BAG OUTDIR [SLAM_PARAMS]: the car's mapping chain on a synthetic bag (make_bag.py), against the truth.
#   twist_estimator + EKF (no PF: dead reckoning, TF odom->base_link) + scan_for_slam + slam_toolbox (sync)
#   -> OUTDIR/log.npy (slam map->base_link next to /truth/odom), OUTDIR/map.{pgm,yaml,posegraph,data}; then slam_eval.py OUTDIR
# Env: DOM (ROS_DOMAIN_ID, parallel runs need different ones), SLAMEXE (async_slam_toolbox_node), RATE.
source /opt/ros/humble/setup.bash; source ~/RC_Car_Codes/install/setup.bash
B=$1; O=$2; P=${3:-~/RC_Car_Codes/src/rc_localization/config/slam_mapping.yaml}
mkdir -p $O; export ROS_DOMAIN_ID=${DOM:-77}
C=~/RC_Car_Codes/src/rc_localization/config; I=~/RC_Car_Codes/install/rc_localization/lib/rc_localization
SL=${SLAMEXE:-sync_slam_toolbox_node}
$I/twist_estimator --ros-args --params-file $C/twist_estimator.yaml -p use_sim_time:=true > $O/twist.log 2>&1 &
$I/ekf_localizer --ros-args --params-file $C/ekf_localizer.yaml -p use_sim_time:=true -p pose_topic:=/no_pose \
  -p max_position_variance:=1.0e9 > $O/ekf.log 2>&1 &
python3 $I/scan_for_slam.py > $O/relay.log 2>&1 &
/opt/ros/humble/lib/slam_toolbox/$SL --ros-args --params-file $P -p use_sim_time:=true > $O/slam.log 2>&1 &
python3 $(dirname $0)/slam_log.py $O/log.npy > $O/logger.log 2>&1 &
LP=$!
sleep 3
ros2 bag play $B --clock 100 --topics /scan /sensors/core /zed2/zed_node/imu/data /truth/odom -r ${RATE:-1} > $O/play.log 2>&1 &
PP=$!
sleep 0.5; ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{header: {frame_id: odom}}' > /dev/null 2>&1
wait $PP; sleep 2
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '$O/map'}}" > /dev/null 2>&1
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$O/map'}" > /dev/null 2>&1
kill -INT $LP; sleep 1
for p in $(jobs -p); do kill -9 $p 2>/dev/null; done
wait
