#!/usr/bin/env bash
# Publish the sensor-rig TF (base_link -> sensors) for the simulator, on the host's ROS 2.
# Uses the same calibration URDF as the simulator, on sim time. The simulator itself publishes
# only odom -> base_link and /clock.
#
#   source /opt/ros/lyrical/setup.bash && scripts/rig_tf.sh [calibration]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALIB_ARGS=()
[ $# -ge 1 ] && CALIB_ARGS=(--calibration "$1")
URDF="$(python3 "$ROOT/scripts/make_rig_urdf.py" "${CALIB_ARGS[@]}")"
export FASTDDS_DEFAULT_PROFILES_FILE="$ROOT/config/fastdds_large_msgs.xml"
exec ros2 run robot_state_publisher robot_state_publisher --ros-args \
    -p robot_description:="$(cat "$URDF")" -p use_sim_time:=true
