#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script installs all the dependencies required to run CV-CUDA benchmarks.
# It uses the /tmp folder to download temporary data and libraries.

# Check CUDA version. Begin by checking if nvcc command exists.
if command -v nvcc >/dev/null 2>&1; then
    # Get CUDA version from nvcc output
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}')

    # Extract major version number
    CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d. -f1)

    # Check major version to determine CUDA version
    if [ "$CUDA_MAJOR_VERSION" -eq 12 ] || [ "$CUDA_MAJOR_VERSION" -eq 13 ]; then
        echo "CUDA $CUDA_MAJOR_VERSION is installed."
    else
        echo "Unknown/Unsupported CUDA version."
        exit 1
    fi
else
    echo "CUDA is not installed."
    exit 1
fi

# Check Python version compatibility
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Detected Python version: $PYTHON_VERSION"

# Validate Python version (benchmarking requires Python 3.10-3.13 per README.md)
if [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 13 ]; then
    echo "Error: Python $PYTHON_VERSION is not compatible with benchmarking (requires Python 3.10-3.13)"
    echo "Please use a compatible Python version"
    exit 1
fi

set -e  # Exit script if any command fails

# Get script directory (before changing to /tmp)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Detect if we need sudo (not needed in Docker containers running as root)
SUDO=""
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
fi

# Install basic packages first.
cd /tmp

# Ensure software-properties-common is installed (usually pre-installed in Docker)
$SUDO apt-get install -y --no-install-recommends software-properties-common 2>/dev/null || true
# Add PPA repository
$SUDO add-apt-repository -y ppa:ubuntu-toolchain-r/test
# Single apt-get update for all subsequent installations
$SUDO apt-get update

$SUDO apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    yasm \
    unzip \
    cmake \
    git

# install g++
$SUDO apt-get install -y --no-install-recommends \
    gcc-11 g++-11 \
    ninja-build

$SUDO update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
$SUDO update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
$SUDO update-alternatives --set gcc /usr/bin/gcc-11
$SUDO update-alternatives --set g++ /usr/bin/g++-11

# Install Python and gtest
$SUDO apt-get install -y --no-install-recommends \
    libgtest-dev \
    libgmock-dev \
    python3-pip \
    ninja-build ccache \
    mlocate && $SUDO updatedb

# Install ffmpeg and other libraries needed for VPF.
# Note: We are not installing either libnv-encode or decode libraries here.
$SUDO apt-get install -y --no-install-recommends \
    ffmpeg \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev\

# Install libssl 1.1.1
cd /tmp
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
$SUDO dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb

# Install NVIDIA NSIGHT Systems 2025.5.1
# Note: Update the build number (currently .1) if a newer build is available
# from https://developer.nvidia.com/nsight-systems
cd /tmp
wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb
$SUDO apt-get install -y \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    libx11-dev \
    libxkbfile-dev \
    /tmp/nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb

echo "export PATH=$PATH:/opt/tensorrt/bin" >> ~/.bashrc

# Install Python packages for benchmarking
echo "Installing Python dependencies for benchmarking..."

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/venv_bench" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv_bench"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv_bench/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install CUDA-specific requirements (includes common requirements)
echo "Installing benchmark dependencies for CUDA $CUDA_MAJOR_VERSION..."
python3 -m pip install -r "$SCRIPT_DIR/requirements_cu${CUDA_MAJOR_VERSION}.txt"

echo ""
echo "Python dependencies installation complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $SCRIPT_DIR/venv_bench/bin/activate"
echo ""

# Done
