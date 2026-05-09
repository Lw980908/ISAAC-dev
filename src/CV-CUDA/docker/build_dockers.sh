#!/bin/bash -ex

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

# Main script used to build the Docker images for the CV-CUDA project
# Usage: ./build_dockers.sh [REGISTRY_PREFIX] [MODE]
# REGISTRY_PREFIX: Optional registry prefix for Docker images (e.g., "myregistry.com/")
#                  If empty or not provided, images will be built locally for native architecture
# MODE: Optional build mode - "multiarch" or "local" (default: "multiarch" if REGISTRY_PREFIX set, "local" otherwise)
#       - "multiarch": Build multi-architecture images (x86_64 + aarch64) and push to registry (requires REGISTRY_PREFIX)
#       - "local": Build for native architecture only and load into local Docker

export VERSION=${VERSION:-5}  # Update version when changing anything in the Dockerfiles
export TEGRA_VERSION=${TEGRA_VERSION:-1} # Update version when changing anything in the Dockerfile.tegra-aarch64-linux.builder

export REGISTRY_PREFIX=${1:-${REGISTRY_PREFIX:-}}
export PYVER=${PYVER:-"py310"}
export MANYLINUX_IMAGE_TAG="2025.10.10-1"

# Python versions for Docker images
export PYTHON_VERSIONS_39_TO_312="3.9 3.10 3.11 3.12"  # For numpy 1 (no Python 3.13 and 3.14 support)
export PYTHON_VERSIONS_39_TO_313="3.9 3.10 3.11 3.12 3.13"  # For Torch 2.8.0 (no Python 3.14 support)
export PYTHON_VERSIONS_310_TO_314="3.10 3.11 3.12 3.13 3.14"  # For Torch 2.9.0 (no Python 3.9 support)

# Parse MODE from command line argument, or auto-detect based on REGISTRY_PREFIX
MODE="${2:-}"
if [[ -z "$MODE" ]]; then
    if [[ -n "$REGISTRY_PREFIX" ]]; then
        MODE="multiarch"
    else
        MODE="local"
    fi
fi

# Validate MODE
if [[ "$MODE" != "multiarch" && "$MODE" != "local" ]]; then
    echo "Error: Unsupported mode '$MODE'. Supported values are: multiarch, local"
    exit 1
fi

# Multiarch mode requires a registry
if [[ "$MODE" == "multiarch" && -z "$REGISTRY_PREFIX" ]]; then
    echo "Error: multiarch mode requires REGISTRY_PREFIX to be set"
    echo "Usage: $0 <REGISTRY_PREFIX> multiarch"
    exit 1
fi

echo "Build mode: $MODE"

# Detect native architecture (used for local builds and context directory naming)
DETECTED_ARCH=$(uname -m)
case "$DETECTED_ARCH" in
    x86_64)
        NATIVE_ARCH="x86_64"
        NATIVE_PLATFORM="linux/amd64"
        ;;
    aarch64|arm64)
        NATIVE_ARCH="aarch64"
        NATIVE_PLATFORM="linux/arm64"
        ;;
    *)
        echo "Error: Unsupported detected architecture '$DETECTED_ARCH'"
        exit 1
        ;;
esac
echo "Native architecture: $NATIVE_ARCH"

# Set platform(s) based on mode
if [[ "$MODE" == "multiarch" ]]; then
    PLATFORMS="linux/amd64,linux/arm64"
    echo "Building for platforms: $PLATFORMS"
else
    PLATFORMS="$NATIVE_PLATFORM"
    echo "Building for platform: $PLATFORMS (local native only)"
fi
export PLATFORMS

# Create isolated build context
BUILD_CONTEXT_DIR="/tmp/cvcuda_build_$$"
echo "Creating isolated build context at: $BUILD_CONTEXT_DIR"
mkdir -p "$BUILD_CONTEXT_DIR"

# Copy all necessary files to the isolated build context
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR"/Dockerfile.* "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/requirements.*.txt "$BUILD_CONTEXT_DIR/" 2>/dev/null || true

# Cleanup function to remove build context on exit
cleanup_build_context() {
    echo "Cleaning up isolated build context: $BUILD_CONTEXT_DIR"
    rm -rf "$BUILD_CONTEXT_DIR"
}
trap cleanup_build_context EXIT

# Determine push/load strategy based on mode
if [[ "$MODE" == "multiarch" ]]; then
    echo "Multiarch mode: pushing images to registry '$REGISTRY_PREFIX'"
    PUSH_OR_LOAD="--push"
else
    echo "Local mode: loading images into local Docker"
    PUSH_OR_LOAD="--load"
fi

# Disable provenance and SBOM attestations to prevent hangs, especially with QEMU emulation
# These features can cause Docker buildx to hang during manifest push phase
ATTESTATION_FLAGS="--provenance=false --sbom=false"

# Set builder name based on mode
if [[ "$MODE" == "multiarch" ]]; then
    BUILDER_NAME="cvcuda_multiarch_builder"
else
    BUILDER_NAME="cvcuda_builder_${NATIVE_ARCH}"
fi

# Check if builder already exists
if docker buildx ls | grep -q "^${BUILDER_NAME}[* ]"; then
    echo "Buildx builder '$BUILDER_NAME' already exists, reusing it"
else
    echo "Creating buildx builder: $BUILDER_NAME"
    docker buildx create --name "$BUILDER_NAME"
fi

# Don't use 'docker buildx use' as it sets global state that conflicts with parallel builds
# Instead, we'll use --builder flag in each build command
docker buildx inspect --bootstrap "$BUILDER_NAME"

# Note about QEMU emulation for multiarch builds
if [[ "$MODE" == "multiarch" ]]; then
    echo "Note: Multi-arch builds will use QEMU emulation for non-native architectures"
    echo "      This may be slower than native builds"
fi

####### BASE IMAGES #######

# Manylinux2_28 with GCC 10
export MANYLINUX_GCC10="${REGISTRY_PREFIX}manylinux2_28_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${MANYLINUX_GCC10}:v${VERSION}} \
    -t ${MANYLINUX_GCC10} -t ${MANYLINUX_GCC10}:v${VERSION} \
    -f Dockerfile.gcc10.deps \
    --build-arg "MANYLINUX_IMAGE_TAG=${MANYLINUX_IMAGE_TAG}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# base with just the toolkit installed (CUDA 12.5.0) on top of Ubuntu 22.04
export CUDA_125="${REGISTRY_PREFIX}cu12.5.0"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${CUDA_125}:v${VERSION}} \
    -t ${CUDA_125} -t ${CUDA_125}:v${VERSION} \
    -f Dockerfile.cuda12.5.0.deps \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# base with just the toolkit installed (CUDA 12.9.0) on top of Ubuntu 22.04
export CUDA_129="${REGISTRY_PREFIX}cu12.9.0"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${CUDA_129}:v${VERSION}} \
    -t ${CUDA_129} -t ${CUDA_129}:v${VERSION} \
    -f Dockerfile.cuda12.9.0.deps \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# base with just the toolkit installed (CUDA 13.0.1) on top of Ubuntu 22.04
export CUDA_1301="${REGISTRY_PREFIX}cu13.0.1"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${CUDA_1301}:v${VERSION}} \
    -t ${CUDA_1301} -t ${CUDA_1301}:v${VERSION} \
    -f Dockerfile.cuda13.0.1.deps \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

####### BUILDER IMAGES #######
# Manylinux-based, various GCC versions
# Dockerfile: Dockerfile.builder.deps

# GCC 10, CUDA 12.5
export BUILDER_CUDA_125="${REGISTRY_PREFIX}builder_cu12.5.0_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${BUILDER_CUDA_125}:v${VERSION}} \
    -t ${BUILDER_CUDA_125} -t ${BUILDER_CUDA_125}:v${VERSION} \
    -f Dockerfile.builder.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=${CUDA_125}:v${VERSION}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# GCC 10, CUDA 12.9
export BUILDER_CUDA_129="${REGISTRY_PREFIX}builder_cu12.9.0_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${BUILDER_CUDA_129}:v${VERSION}} \
    -t ${BUILDER_CUDA_129} -t ${BUILDER_CUDA_129}:v${VERSION} \
    -f Dockerfile.builder.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=${CUDA_129}:v${VERSION}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# GCC 10, CUDA 13.0.1
export BUILDER_CUDA_1301="${REGISTRY_PREFIX}builder_cu13.0.1_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${BUILDER_CUDA_1301}:v${VERSION}} \
    -t ${BUILDER_CUDA_1301} -t ${BUILDER_CUDA_1301}:v${VERSION} \
    -f Dockerfile.builder.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=${CUDA_1301}:v${VERSION}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

####### DEVEL IMAGES #######
# Ubuntu-based, various Python versions

# UBUNTU 22.04, CUDA 12.9, NUMPY 2, TORCH 2.8
export DEVEL_U22_CU129_NUM2_TORCH28="${REGISTRY_PREFIX}devel_u22.04_cu12.9.0_num2_torch2.8.0"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${DEVEL_U22_CU129_NUM2_TORCH28}:v${VERSION}} \
    -t ${DEVEL_U22_CU129_NUM2_TORCH28} -t ${DEVEL_U22_CU129_NUM2_TORCH28}:v${VERSION} \
    -f Dockerfile.devel.deps \
    --build-arg "BASE=nvidia/cuda:12.9.0-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_39_TO_313}" \
    --build-arg "VER_CUDA=12.9.0" \
    --build-arg "VER_NUMPY_MAJOR=2" \
    --build-arg "VER_TORCH=2.8.0" \
    --build-arg "TORCH_ADDITIONAL_INDEX_URL=https://download.pytorch.org/whl/cu129" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# UBUNTU 24.04, CUDA 13.0.1, NUMPY 2, TORCH 2.9
export DEVEL_U24_CU1301_NUM2_TORCH29="${REGISTRY_PREFIX}devel_u24.04_cu13.0.1_num2_torch2.9.0"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${DEVEL_U24_CU1301_NUM2_TORCH29}:v${VERSION}} \
    -t ${DEVEL_U24_CU1301_NUM2_TORCH29} -t ${DEVEL_U24_CU1301_NUM2_TORCH29}:v${VERSION} \
    -f Dockerfile.devel.deps \
    --build-arg "BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_310_TO_314}" \
    --build-arg "VER_CUDA=13.0.1" \
    --build-arg "VER_NUMPY_MAJOR=2" \
    --build-arg "VER_TORCH=2.9.0" \
    --build-arg "TORCH_ADDITIONAL_INDEX_URL=https://download.pytorch.org/whl/cu130" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# UBUNTU 22.04, CUDA 12.5, NUMPY 1, TORCH 2.8
# python 3.13 and 3.14 not supported by numpy 1
export DEVEL_U22_CU125_NUM1_TORCH28="${REGISTRY_PREFIX}devel_u22.04_cu12.5.0_num1_torch2.8.0"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${DEVEL_U22_CU125_NUM1_TORCH28}:v${VERSION}} \
    -t ${DEVEL_U22_CU125_NUM1_TORCH28} -t ${DEVEL_U22_CU125_NUM1_TORCH28}:v${VERSION} \
    -f Dockerfile.devel.deps \
    --build-arg "BASE=nvidia/cuda:12.5.0-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_39_TO_312}" \
    --build-arg "VER_CUDA=12.5.0" \
    --build-arg "VER_NUMPY_MAJOR=1" \
    --build-arg "VER_TORCH=2.8.0" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"


docker buildx stop "$BUILDER_NAME"
docker buildx rm "$BUILDER_NAME"
