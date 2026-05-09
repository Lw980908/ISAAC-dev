#!/usr/bin/env bash
# Isaac ROS runtime environment setup
# Source this file before running Isaac ROS nodes

ISAAC_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAAC_ROS_WS="${ISAAC_WS}"

# Source ROS 2 and workspace
source /opt/ros/humble/setup.bash
source "${ISAAC_WS}/install/setup.bash"

# Add all GXF library paths
for dir in "${ISAAC_WS}"/install/*/share/*/gxf/lib; do
  if [[ -d "$dir" ]]; then
    export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"
  fi
done

# CV-CUDA paths (if installed)
if [[ -d "/opt/nvidia/cvcuda" ]]; then
  export LD_LIBRARY_PATH="/opt/nvidia/cvcuda/lib/aarch64-linux-gnu:/opt/nvidia/cvcuda/lib:${LD_LIBRARY_PATH}"
fi

echo "Isaac ROS environment ready."
