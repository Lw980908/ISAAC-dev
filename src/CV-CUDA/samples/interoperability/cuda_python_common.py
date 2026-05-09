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

from __future__ import annotations

import cuda.bindings.runtime as cudart
import numpy as np


# docs_tag: begin_cuda_buffer
class CudaBuffer:
    """Wrapper for CUDA memory buffer that implements __cuda_array_interface__."""

    def __init__(self, shape, dtype, ptr=None):
        """Initialize CUDA buffer.

        Args:
            shape: tuple of dimensions
            dtype: numpy dtype
            ptr: CUDA device pointer (if None, allocates new memory)
        """
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(shape)) * self.dtype.itemsize

        if ptr is None:
            err, self.ptr = cudart.cudaMalloc(self.size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed: {err}")
            self.owns_memory = True
        else:
            self.ptr = ptr
            self.owns_memory = False

    @staticmethod
    def from_cuda(cuda_obj) -> "CudaBuffer":
        """Create from __cuda_array_interface__ or object that implements it."""
        if hasattr(cuda_obj, "__cuda_array_interface__"):
            cuda_array_interface = cuda_obj.__cuda_array_interface__
        else:
            cuda_array_interface = cuda_obj
        return CudaBuffer(
            shape=cuda_array_interface["shape"],
            dtype=np.dtype(cuda_array_interface["typestr"]),
            ptr=cuda_array_interface["data"][0],
        )

    @property
    def __cuda_array_interface__(self):
        """CUDA Array Interface for zero-copy interop."""
        return {
            "version": 3,
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (int(self.ptr), False),
            "strides": None,
        }

    def __del__(self):
        """Free CUDA memory if we own it."""
        if self.owns_memory and hasattr(self, "ptr"):
            cudart.cudaFree(self.ptr)


# docs_tag: end_cuda_buffer

# docs_tag: being_cuda_memcpy_h2d
def cuda_memcpy_h2d(
    host_array: np.ndarray,
    device_array: int | dict | object,
) -> None:
    """
    Copy host array to device array.

    Args:
        host_array: Host array to copy.
        device_array: Device array to copy to, __cuda_array_interface__ or object with interface.
    """
    if hasattr(device_array, "__cuda_array_interface__"):
        device_array = device_array.__cuda_array_interface__["data"][0]
    elif isinstance(device_array, dict) and "data" in device_array:
        device_array = device_array["data"][0]
    elif not isinstance(device_array, int):
        raise ValueError("Invalid device array")
    (err,) = cudart.cudaMemcpy(
        device_array,
        host_array.ctypes.data,
        host_array.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy failed: {err}")


# docs_tag: end_cuda_memcpy_h2d

# docs_tag: begin_cuda_memcpy_d2h
def cuda_memcpy_d2h(
    device_array: int | dict | object,
    host_array: np.ndarray,
) -> None:
    """
    Copy device array to host array.

    Args:
        device_array: Device array to copy from, __cuda_array_interface__ or object with interface.
        host_array: Host array to copy to.
    """
    if hasattr(device_array, "__cuda_array_interface__"):
        device_array = device_array.__cuda_array_interface__["data"][0]
    elif isinstance(device_array, dict) and "data" in device_array:
        device_array = device_array["data"][0]
    elif not isinstance(device_array, int):
        raise ValueError("Invalid device array")
    (err,) = cudart.cudaMemcpy(
        host_array.ctypes.data,
        device_array,
        host_array.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy failed: {err}")


# docs_tag: end_cuda_memcpy_d2h
