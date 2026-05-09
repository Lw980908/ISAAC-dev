[//]: # "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."

# CV-CUDA Docker Infrastructure

This directory contains the Docker infrastructure for building and developing CV-CUDA across multiple environments, architectures, and dependency combinations.

## Overview

The Docker setup is organized into two main categories of images:
1. **Builder Images** - Manylinux-based images for building redistributable packages
2. **Development Images** - Ubuntu-based images with full development/test environments

All images are **multi-architecture manifests** supporting:
- **x86_64** (AMD64)
- **aarch64** (ARM64)

Docker automatically selects the appropriate architecture when pulling images.

## Builder Images

Combines manylinux base with CUDA toolkit for building CV-CUDA packages.

| Image Name | GCC Version | CUDA Version | Base Image |  Purpose |
|------------|-------------|--------------|----------------|---------|
| `builder_cu12.5.0_gcc10` | 10 | 12.5.0 | `manylinux_2_28` | CUDA 12.5 builds for CV-CUDA packages (multi-arch) |
| `builder_cu12.9.0_gcc10` | 10 | 12.9.0 | `manylinux_2_28` | CUDA 12.9 builds for CV-CUDA packages (multi-arch) |
| `builder_cu13.0.1_gcc10` | 10 | 13.0.1 | `manylinux_2_28` | CUDA 13.0 builds for CV-CUDA packages (multi-arch) |

### Build Dependencies

| Image Name | Base Image | Dockerfile | Purpose |
|------------|------------|------------|---------|
| manylinux2_28_gcc${GCC_VER} | ManyLinux 2_28 (gcc 10) | Dockerfile.gcc${GCC_VER}.deps | Gcc base images, adding gcc to ManyLinux (multi-arch) |
| cu${CUDA_VER} | Ubuntu 22.04 | Dockerfile.cuda${CU_VER}.deps | CUDA base images, adding cuda toolkit to Ubuntu (multi-arch) |
| builder_cu${CUDA_VER}_gcc${GCC_VER} | ManyLinux 2_28 (gcc 10) | Dockerfile.builder.deps | Builder images, combining the Gcc base images with CUDA copied from the CUDA base images (multi-arch) |



```text
┌─────────────────┐    ┌─────────────────┐
│   ManyLinux     │    │   Ubuntu 22.04  │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          │ + GCC                │ + CUDA Toolkit
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ GCC Base Images │    │ CUDA Base Images│
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          │ BASE                 │ COPY CUDA binaries
          └──────────┬───────────┘
                     │
                     │ Combine
                     ▼
           ┌─────────────────┐
           │ Builder Images  │
           └─────────────────┘
```



### Builder Image Features

All builder images include:
- **CMake 3.24.3** - Modern build system
- **Python Support** - Multiple Python versions (3.9-3.14) from ManyLinux
- **Documentation Tools** - Sphinx 7.4.7, sphinx_rtd_theme, breathe
- **Development Tools** - patchelf 0.17.2, setuptools, wheel, clang 14.0
- **CUDA Integration** - Full CUDA toolkit from base CUDA images
- **Git Support** - git-lfs for large file handling

## Development Images

Full Ubuntu-based environments with multiple Python versions, Numpy and Torch for development or testing purposes.

We leverage [NVIDIA-maintained base images](https://hub.docker.com/r/nvidia/cuda/tags) `nvidia/cuda:${CUDA_VER}$-devel-ubuntu${UB_VER}$`.

Dockerfile:  `Dockerfile.devel.deps`

The build script `build_dockers.sh` creates several development image variants:

| Image | Base Image | CUDA | NumPy | PyTorch | Python Versions |
|-------|------------|------|-------|---------|-----------------|
| `devel_u22.04_cu12.9.0_num2_torch2.8.0` | nvidia/cuda:12.9.0-devel-ubuntu22.04 | 12.9.0 | 2.x (2.0.2-2.3.3) | 2.8.0 | 3.9-3.13 (multi-arch) |
| `devel_u22.04_cu12.5.0_num1_torch2.8.0` | nvidia/cuda:12.5.0-devel-ubuntu22.04 | 12.5.0 | 1.26.4 | 2.8.0 | 3.9-3.13 (multi-arch) |
| `devel_u24.04_cu13.0.1_num2_torch2.9.0` | nvidia/cuda:13.0.1-devel-ubuntu24.04 | 13.0.1 | 2.x (2.2.6-2.3.3) | 2.9.0 | 3.10-3.14 (multi-arch) |

*Note: PyTorch 2.8.0 supports Python 3.9-3.13, PyTorch 2.9.0 supports Python 3.10-3.14

**Key Features:**
- **Python Versions:** 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
- **Development Tools:** CMake 3.24.3, build-essential, clang-14, ninja-build
- **Testing:** Google Test/Mock, pytest
- **ML Frameworks:** PyTorch, CuPy (CUDA-specific versions)
- **Documentation:** Doxygen, Sphinx ecosystem
- **Version Control:** git, git-lfs, pre-commit

### Python Requirements

| File | Purpose | Usage |
|------|---------|-------|
| `requirements.sys_python.txt` | System Python packages, installed only for the system Python version | Documentation and build tools |
| `requirements.no_torch_no_numpy.txt` | Development/test packages, installed for each Python version | Testing, building (excludes Torch and Numpy) |
| `requirements.numpy1.txt` | NumPy 1.x requirements with Python version constraints | NumPy 1.26.4 for Python 3.9-3.12 (selected via VER_NUMPY_MAJOR=1) |
| `requirements.numpy2.txt` | NumPy 2.x requirements with Python version constraints | NumPy 2.x versions for Python 3.9-3.14 (selected via VER_NUMPY_MAJOR=2) |

### Build Script

**`build_dockers.sh`** - Main orchestration script for building all Docker images.

#### Usage
```bash
# Build locally for native architecture only (default)
./build_dockers.sh

# Explicitly force local build mode (native architecture only)
./build_dockers.sh "" local

# Build and push multi-arch images (x86_64 + aarch64) to registry
./build_dockers.sh $REGISTRY_PREFIX multiarch
```
where $REGISTRY_PREFIX should be set to your remote registry.

**Modes:**
- `local` - Build for native architecture only, load into local Docker (default when no registry)
- `multiarch` - Build for both x86_64 and aarch64, push to registry (default when registry provided)

### Using Development Images
```bash
# Run development container (automatically selects correct architecture)
docker run -it --gpus all devel_u22.04_cu12.5.0_num1_torch2.8.0:v2

# Mount source code for development
docker run -it --gpus all \
  -v /path/to/cvcuda:/workspace \
  devel_u22.04_cu12.5.0_num1_torch2.8.0:v2
```

### Using Builder Images for Package Creation
```bash
# Use builder for creating wheels (automatically selects correct architecture)
docker run -it --gpus all \
  -v /path/to/cvcuda:/workspace \
  builder_cu12.5.0_gcc10:v1
```

## Maintenance Notes

### Updating image Versions

Increment `VERSION` variable in `build_dockers.sh` when changing Dockerfiles.
Run build script to create new image versions

### Adding New CUDA Versions
1. Create new `Dockerfile.cuda{version}.deps` with architecture detection logic (see existing files for pattern)
   - Use `dpkg --print-architecture` to detect amd64 vs arm64
   - Download appropriate CUDA installer (`linux.run` for x86_64, `linux_sbsa.run` for aarch64)
2. Add corresponding sections in `build_dockers.sh`
3. Update development image variants as needed

### Adding New Python Versions

For builder images, Python versions available come directly from the base ManyLinux.

For development images, update the build arguments to docker buildx in `build_dockers.sh`: `--build-arg "PYTHON_VERSIONS=3.9 3.10 3.11 3.12 3.13 3.14"`


## Troubleshooting

### Build Failures
- Check Docker buildx is installed and available
- Ensure sufficient disk space for multi-stage builds
- Verify network connectivity for downloading CUDA installers

### Cache Issues
- Use `docker system prune` to clear build cache
- Remove and recreate buildx builder: `docker buildx rm cvcuda_multiarch_builder` (or `cvcuda_builder_x86_64`/`cvcuda_builder_aarch64` for local builds)

### Registry Authentication
- Ensure proper authentication to registry before using `REGISTRY_PREFIX`
- Use `docker login` for private registries
