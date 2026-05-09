#!/usr/bin/env bash
set -eo pipefail

# Isaac ROS build helper (JetPack 6.x + ROS 2 Humble)
# Usage:
#   ./build_isaac.sh
#   ISAAC_WS=/path/to/ISAAC ./build_isaac.sh
#   BUILD_LFS=0 BUILD_CVCUDA=0 ./build_isaac.sh  # Skip LFS and CV-CUDA
#
# Environment variables:
#   BUILD_LFS=1|0     - Pull Git LFS objects (default: 1)
#   BUILD_CVCUDA=1|0  - Build and install CV-CUDA (default: 1)
#   CVCUDA_ONLY=1     - Build CV-CUDA only and exit (default: 0)
#   BUILD_MODELS_INSTALL=1|0 - Build/download model assets (*_models_install packages) (default: 0)

ISAAC_WS="${ISAAC_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SRC_DIR="${ISAAC_WS}/src"
BUILD_LFS="${BUILD_LFS:-1}"
BUILD_CVCUDA="${BUILD_CVCUDA:-1}"
BUILD_MODELS_INSTALL="${BUILD_MODELS_INSTALL:-0}"

# Some Isaac ROS packages use isaac_ros_common asset install helpers which expect this env var.
export ISAAC_ROS_WS="${ISAAC_WS}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "ERROR: ${SRC_DIR} not found."
  echo "Set ISAAC_WS to your workspace root."
  exit 1
fi

echo "[1/7] Checking required repos..."
for repo in isaac_ros_common isaac_ros_compression isaac_ros_nitros negotiated isaac_ros_rtsp_server; do
  if [[ ! -d "${SRC_DIR}/${repo}" ]]; then
    echo "ERROR: missing repo ${SRC_DIR}/${repo}"
    exit 1
  fi
done

if [[ ! -d "${SRC_DIR}/isaac_ros_argus_camera" ]]; then
  echo "WARN: missing optional repo ${SRC_DIR}/isaac_ros_argus_camera (Argus camera support disabled)"
fi

if [[ "${BUILD_CVCUDA}" == "1" ]]; then
  if [[ ! -d "${SRC_DIR}/CV-CUDA" ]]; then
    echo "ERROR: missing repo ${SRC_DIR}/CV-CUDA (required when BUILD_CVCUDA=1)"
    exit 1
  fi
fi

echo "[2/7] Installing build dependencies..."
sudo apt update
sudo apt install -y \
  cmake ninja-build build-essential \
  python3-dev python3-pip

GIT_GITHUB_MIRROR="${GIT_GITHUB_MIRROR:-}"
if [[ -n "${GIT_GITHUB_MIRROR}" ]]; then
  git config --global url."${GIT_GITHUB_MIRROR}".insteadOf "https://github.com/"
fi

if [[ "${BUILD_LFS}" == "1" ]]; then
  sudo apt install -y git-lfs
fi

if [[ "${BUILD_CVCUDA}" == "1" ]]; then
  sudo apt install -y patchelf libtbb-dev libopencv-dev
fi

if [[ "${BUILD_LFS}" == "1" ]]; then
  echo "[3/7] Pulling Git LFS objects..."
  git lfs install
  for repo in isaac_ros_nitros isaac_ros_compression; do
    if [[ -d "${SRC_DIR}/${repo}/.git" ]]; then
      (cd "${SRC_DIR}/${repo}" && git lfs pull)
    fi
  done
else
  echo "[3/7] Skipping Git LFS pull (BUILD_LFS=0)"
fi

if [[ "${BUILD_CVCUDA}" == "1" ]]; then
  echo "[4/7] Building CV-CUDA (C++ only, no Python)..."
  export CUDA_HOME=/usr/local/cuda
  export CUDACXX=/usr/local/cuda/bin/nvcc

  mkdir -p "${SRC_DIR}/CV-CUDA/build"
  (
    cd "${SRC_DIR}/CV-CUDA/build"
    cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/nvidia/cvcuda \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DBUILD_PYTHON=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_TESTS_CPP=OFF \
      -DBUILD_TESTS_PYTHON=OFF \
      -DBUILD_TESTS_WHEELS=OFF
    ninja -j"$(nproc)"
    sudo ninja install
  )

  if [[ "${CVCUDA_ONLY:-0}" == "1" ]]; then
    echo "CV-CUDA build complete (CVCUDA_ONLY=1)."
    exit 0
  fi
else
  echo "[4/7] Skipping CV-CUDA build (BUILD_CVCUDA=0)"
fi

echo "[5/7] Ensuring live555 is available..."
if sudo apt install -y liblivemedia-dev libgroupsock-dev libusageenvironment-dev libbasicusageenvironment-dev; then
  echo "[5/7] live555 dev packages installed."
else
  if [[ -d "${SRC_DIR}/live555" ]]; then
    if [[ ! -f "${SRC_DIR}/live555/liveMedia/libliveMedia.so" ]] && \
       [[ ! -f "${SRC_DIR}/live555/liveMedia/libliveMedia.a" ]]; then
      echo "[5/7] Building live555 from source (shared libs)..."
      (
        cd "${SRC_DIR}/live555"
        ./genMakefiles linux-with-shared-libraries
        make -j"$(nproc)"
        sudo make install
        sudo ldconfig || true
      )
    else
      echo "[5/7] live555 already built, skipping."
    fi
  else
    echo "ERROR: live555 not found and apt install failed."
    exit 1
  fi
fi

echo "[6/7] Installing ROS deps (rosdep)..."
# Source ROS 2 setup (may use unbound variables, which is OK)
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH || true
source /opt/ros/humble/setup.bash || true
rosdep install --from-paths "${SRC_DIR}" --ignore-src -r -y \
  --skip-keys "posix_ipc cuda_python negotiated"

echo "[7/7] Building Isaac ROS workspace..."
if [[ "${BUILD_CVCUDA}" == "1" ]]; then
  export CMAKE_PREFIX_PATH=/opt/nvidia/cvcuda:${CMAKE_PREFIX_PATH:-}
  export LD_LIBRARY_PATH=/opt/nvidia/cvcuda/lib:${LD_LIBRARY_PATH:-}
fi

cd "${ISAAC_WS}"
rm -rf build log install

# Clean any CMake FetchContent cache that might be causing issues
find "${SRC_DIR}" -type d -name "_deps" -exec rm -rf {} + 2>/dev/null || true

# Skip CV-CUDA directory from colcon build (it's installed via ninja install, not as ROS package)
if [[ -d "${SRC_DIR}/CV-CUDA" ]]; then
  touch "${SRC_DIR}/CV-CUDA/COLCON_IGNORE"
  echo "Note: Skipping CV-CUDA from colcon build (already installed to /opt/nvidia/cvcuda)"
fi

# Set CV-CUDA paths for CMake to find (required for custom_nitros_dnn_image_encoder)
if [[ "${BUILD_CVCUDA}" == "1" ]] || [[ -d "/opt/nvidia/cvcuda" ]]; then
  export CMAKE_PREFIX_PATH=/opt/nvidia/cvcuda:${CMAKE_PREFIX_PATH:-}
  export LD_LIBRARY_PATH=/opt/nvidia/cvcuda/lib/aarch64-linux-gnu:/opt/nvidia/cvcuda/lib:${LD_LIBRARY_PATH:-}
  export PKG_CONFIG_PATH=/opt/nvidia/cvcuda/lib/aarch64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}
  export CXXFLAGS="-I/opt/nvidia/cvcuda/include ${CXXFLAGS:-}"
  
  # Ensure CV-CUDA libraries are in the system library path
  if [[ -f "/opt/nvidia/cvcuda/etc/ld.so.conf.d/cvcuda0.conf" ]]; then
    sudo ldconfig || true
  fi
fi

COLCON_SKIP_ARGS=()
if [[ "${BUILD_MODELS_INSTALL}" != "1" ]]; then
  for pkg in isaac_ros_grounding_dino_models_install isaac_ros_rtdetr_models_install isaac_ros_peoplenet_models_install; do
    if [[ -d "${SRC_DIR}/isaac_ros_object_detection/${pkg}" ]]; then
      COLCON_SKIP_ARGS+=(--packages-skip "${pkg}")
    fi
  done
fi

colcon build --symlink-install \
  "${COLCON_SKIP_ARGS[@]}" \
  --cmake-args \
    -DCMAKE_DEVICE=arm64 \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCVCUDA_DIR=/opt/nvidia/cvcuda \
    -DCMAKE_PREFIX_PATH=/opt/nvidia/cvcuda \
    -DCMAKE_CXX_FLAGS="-I/opt/nvidia/cvcuda/include"

# Generate runtime environment script
ENV_SCRIPT="${ISAAC_WS}/env_setup.bash"
cat > "${ENV_SCRIPT}" << 'ENVEOF'
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
ENVEOF
chmod +x "${ENV_SCRIPT}"

echo ""
echo "Done. To run Isaac ROS nodes, source the environment script:"
echo "  source ${ISAAC_WS}/env_setup.bash"
echo ""
echo "Or add to ~/.bashrc for permanent setup:"
echo "  echo 'source ${ISAAC_WS}/env_setup.bash' >> ~/.bashrc"

