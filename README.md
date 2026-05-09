Isaac ROS build guide (JetPack 6.2 + ROS 2 Humble)
==================================================

This workspace expects the following repos under `src/`:
- `CV-CUDA` (built with BUILD_PYTHON=OFF)
- `isaac_ros_argus_camera`
- `isaac_ros_common`
- `isaac_ros_compression`
- `isaac_ros_nitros`
- `negotiated`

Get sources (git clone)
-----------------------

From the workspace root:
```
mkdir -p /home/nvidia/ISAAC/src
cd /home/nvidia/ISAAC/src

git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_argus_camera.git
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_compression.git
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git
git clone https://github.com/osrf/negotiated.git

git clone https://github.com/CVCUDA/CV-CUDA.git
```

Quick start
-----------

1) From the workspace root:
```
cd /home/nvidia/ISAAC
```

2) Build CV-CUDA first (C++ only, optional):
```
CVCUDA_ONLY=1 ./build_isaac.sh
```

3) Run the full build helper:
```
./build_isaac.sh
```

Optional build steps
--------------------

You can skip LFS pull or CV-CUDA build if already done:

- Skip Git LFS pull (if binaries already downloaded):
  ```
  BUILD_LFS=0 ./build_isaac.sh
  ```

- Skip CV-CUDA build (if already installed):
  ```
  BUILD_CVCUDA=0 ./build_isaac.sh
  ```

- Skip both:
  ```
  BUILD_LFS=0 BUILD_CVCUDA=0 ./build_isaac.sh
  ```

What the script does
--------------------

- Installs build dependencies (`cmake`, `ninja`, etc.)
- Optionally pulls Git LFS objects for GXF prebuilt libraries (`git lfs pull`, controlled by `BUILD_LFS`)
- Optionally builds and installs CV-CUDA to `/opt/nvidia/cvcuda` (C++ only, controlled by `BUILD_CVCUDA`)
- Runs `rosdep` with known skip keys
- Builds the workspace with:
  - `-DCMAKE_DEVICE=arm64`
  - `-DCUDAToolkit_ROOT=/usr/local/cuda`

After build
-----------

```
source /home/nvidia/ISAAC/install/setup.bash
```

Troubleshooting
---------------

### Missing GXF packages (gxf_isaac_tensorops / gxf_isaac_rectify)

Symptoms (launch failure):
```
Component constructor threw an exception: package 'gxf_isaac_tensorops' not found
```
or
```
Component constructor threw an exception: package 'gxf_isaac_rectify' not found
```

Cause:
- These GXF extensions are **prebuilt Debian packages** from the Isaac ROS apt repo.
- They are **not** built from source in this workspace, so `colcon build` alone won't create them.

Fix:
1. Add the Isaac ROS apt repo (once):
   ```bash
   wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo apt-key add -
   echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" | sudo tee -a /etc/apt/sources.list
   sudo apt update
   ```
2. Install missing packages:
   ```bash
   sudo apt install ros-humble-gxf-isaac-tensorops ros-humble-gxf-isaac-rectify ros-humble-isaac-ros-stereo-image-proc
   ```
3. Rebuild the workspace after installation:
   ```bash
   cd /home/nvidia/ISAAC
   ./build_isaac.sh
   ```

### CMake FetchContent download failures

If you see errors like:
```
Build step for metadata failed: 2
```

This is usually a network issue. Try:

1. **Clean CMake cache and retry:**
   ```bash
   cd /home/nvidia/ISAAC
   find src -type d -name "_deps" -exec rm -rf {} + 2>/dev/null || true
   rm -rf build log install
   ./build_isaac.sh
   ```

2. **Configure Git proxy (if behind firewall):**
   ```bash
   git config --global http.proxy http://your-proxy:port
   git config --global https.proxy https://your-proxy:port
   ```

3. **Use GitHub mirror (if in China):**
   ```bash
   git config --global url."https://mirror.ghproxy.com/https://github.com/".insteadOf "https://github.com/"
   ```

4. **Manual retry with verbose output:**
   ```bash
   cd /home/nvidia/ISAAC
   colcon build --symlink-install --event-handlers console_direct+ \
     --cmake-args -DCMAKE_DEVICE=arm64 -DCUDAToolkit_ROOT=/usr/local/cuda
   ```

Notes
-----

- If you use a different workspace path, set:
  - `ISAAC_WS=/your/path ./build_isaac.sh`
- If `git lfs pull` is slow, let it finish. It downloads large `.so` files
  required by Isaac ROS (GXF runtime libraries).
