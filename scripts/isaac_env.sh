#!/usr/bin/env bash
# Environment for running the pip-installed Isaac Sim with its bundled ROS 2 (Jazzy) libs.
#
# The host runs ROS 2 Lyrical, which Isaac Sim 6.1 does not ship libs for. We therefore run
# Isaac on its internal Jazzy libs and rely on DDS wire compatibility (Fast DDS on both sides)
# to talk to Lyrical nodes. Never source /opt/ros/* in the same shell: mixing the host's libs
# with the bundled ones breaks symbol resolution at process start.
#
# Usage: source scripts/isaac_env.sh

ISAACSIM_VENV="${ISAACSIM_VENV:-$HOME/isaacsim/.venv}"
ISAACSIM_PKG="$ISAACSIM_VENV/lib/python3.12/site-packages/isaacsim"
FARM_SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Scrub anything a sourced ROS install may have left behind.
for v in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH ROS_PYTHON_VERSION ROS_VERSION; do
    unset "$v"
done
LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/opt/ros' | paste -sd: -)"

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ISAACSIM_PKG/exts/isaacsim.ros2.core/jazzy/lib"
export PYTHONPATH="$FARM_SIM_ROOT"

# shellcheck disable=SC1091
source "$ISAACSIM_VENV/bin/activate"

# Larger UDP socket buffers for point clouds / images (see config/fastdds_large_msgs.xml).
export FASTRTPS_DEFAULT_PROFILES_FILE="$FARM_SIM_ROOT/config/fastdds_large_msgs.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$FARM_SIM_ROOT/config/fastdds_large_msgs.xml"
