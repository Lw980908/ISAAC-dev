#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script installs all the dependencies required to run CV-CUDA interoperability samples.
# It detects the CUDA version and installs appropriate packages.

# Check CUDA version
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk -F'release ' '{print $2}' | cut -d',' -f1)
    CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d. -f1)

    if [ "$CUDA_MAJOR_VERSION" -eq 12 ] || [ "$CUDA_MAJOR_VERSION" -eq 13 ]; then
        echo "CUDA $CUDA_MAJOR_VERSION is installed."
    else
        echo "Unknown/Unsupported CUDA version."
        exit 1
    fi
else
    echo "Error: CUDA is not installed."
    exit 1
fi

set -e  # Exit script if any command fails

echo "Installing Python dependencies for CV-CUDA interoperability samples..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/venv_samples" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv_samples"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv_samples/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install interoperability dependencies for the detected CUDA version
echo "Installing interoperability dependencies for CUDA $CUDA_MAJOR_VERSION..."
python3 -m pip install -r "$SCRIPT_DIR/requirements_interop_cu${CUDA_MAJOR_VERSION}.txt"

echo ""
echo "Interoperability dependencies installation complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $SCRIPT_DIR/venv_samples/bin/activate"
echo ""
echo "Then you can run interoperability samples:"
echo "  python3 interoperability/pytorch_interop.py"
echo "  python3 interoperability/cupy_interop.py"
echo ""
