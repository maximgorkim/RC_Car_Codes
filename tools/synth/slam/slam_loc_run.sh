#!/bin/bash
# slam_loc_run.sh BAG MAPDIR OUTDIR "x,y,yaw" [SLAM_PARAMS]: slam_toolbox LOCALIZATION mode on a synthetic bag,
# against the truth. The same chain as mapping (slam_run.sh) but localization_slam_toolbox_node on the
# serialized map MAPDIR/map.{posegraph,data}; x,y,yaw = the car's start pose in that map's frame.
#   twist_estimator + EKF (no PF: dead reckoning, TF odom->base_link) + scan_for_slam + slam_toolbox (localization)
#   -> OUTDIR/log.npy (map->base_link next to /truth/odom), OUTDIR/cpu.txt (slam node CPU seconds, bag seconds)
# Env: DOM (ROS_DOMAIN_ID, parallel runs need different ones), RATE, SET ("name:=value ..." extra slam params),
#      SHIFT (scan_for_slam stamp_shift [s]).
source /opt/ros/humble/setup.bash; source ~/RC_Car_Codes/install/setup.bash
B=$1; M=$2; O=$3; IFS=, read SX SY SA <<< "$4"; P=${5:-~/RC_Car_Codes/src/rc_localization/config/slam_mapping.yaml}
mkdir -p $O; export ROS_DOMAIN_ID=${DOM:-77}
C=~/RC_Car_Codes/src/rc_localization/config; I=~/RC_Car_Codes/install/rc_localization/lib/rc_localization
EXTRA=""; for kv in $SET; do EXTRA="$EXTRA -p $kv"; done
$I/twist_estimator --ros-args --params-file $C/twist_estimator.yaml -p use_sim_time:=true > $O/twist.log 2>&1 &
$I/ekf_localizer --ros-args --params-file $C/ekf_localizer.yaml -p use_sim_time:=true -p pose_topic:=/no_pose \
  -p max_position_variance:=1.0e9 > $O/ekf.log 2>&1 &
python3 $I/scan_for_slam.py --ros-args -p stamp_shift:=${SHIFT:-0.0} > $O/relay.log 2>&1 &
/opt/ros/humble/lib/slam_toolbox/localization_slam_toolbox_node --ros-args --params-file $P -p use_sim_time:=true \
  -p mode:=localization -p map_file_name:=$M/map -p "map_start_pose:=[$SX, $SY, $SA]" $EXTRA > $O/slam.log 2>&1 &
SP=$!
python3 $(dirname $0)/slam_log.py $O/log.npy > $O/logger.log 2>&1 &
LP=$!
sleep 3
# paused start: the EKF (odom origin) and map_start_pose must both be the car's pose at the bag's start
ros2 bag play $B --clock 100 --start-paused --topics /scan /sensors/core /zed2/zed_node/imu/data /truth/odom -r ${RATE:-1} > $O/play.log 2>&1 &
PP=$!
# slam_toolbox listens to /initialpose too: the start pose for both, so map -> odom starts as identity
QZ=$(python3 -c "import math; print(math.sin($SA / 2), math.cos($SA / 2))")
sleep 2; ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: odom}, pose: {pose: \
  {position: {x: $SX, y: $SY}, orientation: {z: ${QZ% *}, w: ${QZ#* }}}}}" > /dev/null 2>&1
sleep 0.5; ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume > /dev/null 2>&1
T0=$(date +%s.%N); C0=$(awk '{print $14+$15}' /proc/$SP/stat)
wait $PP; sleep 1
T1=$(date +%s.%N); C1=$(awk '{print $14+$15}' /proc/$SP/stat)
echo "$(echo "($C1 - $C0) / $(getconf CLK_TCK)" | bc -l) $(echo "$T1 - $T0" | bc -l)" > $O/cpu.txt
kill -INT $LP; sleep 1
for p in $(jobs -p); do kill -9 $p 2>/dev/null; done
wait
