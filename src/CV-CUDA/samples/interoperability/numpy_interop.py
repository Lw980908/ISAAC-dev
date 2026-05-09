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
"""
Example showcasing how to transfer NumPy arrays to GPU and use with CV-CUDA.

This example demonstrates four different methods to transfer NumPy data to GPU:
1. CUDA Python (cuda-python) - See cuda_python_interop.py for more details
2. PyTorch (torch) - See pytorch_interop.py for more details
3. CuPy - See cupy_interop.py for more details
4. PyCUDA - See pycuda_interop.py for more details
"""

# docs_tag: begin_imports
import numpy as np
import cvcuda

# docs_tag: end_imports


def numpy_to_cvcuda_via_cuda_python():
    """
    Transfer NumPy to GPU using CUDA Python, then create CV-CUDA tensor.

    For more details on CUDA Python interop, see cuda_python_interop_1.py
    """

    from cuda_python_interop_1 import CudaBuffer, cuda_memcpy_d2h, cuda_memcpy_h2d

    # docs_tag: begin_numpy_cuda_python
    numpy_array = np.random.randn(10, 10).astype(np.float32)
    cuda_buffer = CudaBuffer(numpy_array.shape, numpy_array.dtype)
    cuda_memcpy_h2d(numpy_array, cuda_buffer.ptr)

    cvcuda_tensor = cvcuda.as_tensor(cuda_buffer)
    # docs_tag: end_numpy_cuda_python

    # docs_tag: begin_numpy_cuda_python_verify
    # Copy back to NumPy and verify
    result_array = np.empty_like(numpy_array)

    cuda_memcpy_d2h(
        cvcuda_tensor.cuda().__cuda_array_interface__["data"][0], result_array
    )

    assert np.allclose(numpy_array, result_array)
    # docs_tag: end_numpy_cuda_python_verify


def numpy_to_cvcuda_via_torch():
    """Transfer NumPy to GPU using PyTorch, then create CV-CUDA tensor.

    For more details on PyTorch interop, see pytorch_interop.py
    """
    import torch

    # docs_tag: begin_numpy_torch
    numpy_array = np.random.randn(10, 10).astype(np.float32)

    torch_tensor = torch.from_numpy(numpy_array).cuda()

    cvcuda_tensor = cvcuda.as_tensor(torch_tensor)
    # docs_tag: end_numpy_torch

    # docs_tag: begin_numpy_torch_verify
    result_torch = torch.as_tensor(cvcuda_tensor.cuda()).clone()
    result_array = result_torch.cpu().numpy()

    assert np.allclose(numpy_array, result_array)
    # docs_tag: end_numpy_torch_verify


def numpy_to_cvcuda_via_cupy():
    """Transfer NumPy to GPU using CuPy, then create CV-CUDA tensor.

    For more details on CuPy interop, see cupy_interop.py
    """
    import cupy as cp

    # docs_tag: begin_numpy_cupy
    numpy_array = np.random.randn(10, 10).astype(np.float32)

    cupy_array = cp.asarray(numpy_array)

    cvcuda_tensor = cvcuda.as_tensor(cupy_array)
    # docs_tag: end_numpy_cupy

    # docs_tag: begin_numpy_cupy_verify
    result_cupy = cp.asarray(cvcuda_tensor.cuda())
    result_array = cp.asnumpy(result_cupy)

    assert np.allclose(numpy_array, result_array)
    # docs_tag: end_numpy_cupy_verify


def numpy_to_cvcuda_via_pycuda():
    """Transfer NumPy to GPU using PyCUDA, then create CV-CUDA tensor.

    For more details on PyCUDA interop, see pycuda_interop.py
    """
    import pycuda.autoinit  # noqa: F401
    import pycuda.gpuarray as gpuarray

    # docs_tag: begin_numpy_pycuda
    numpy_array = np.random.randn(10, 10).astype(np.float32)

    pycuda_array = gpuarray.to_gpu(numpy_array)

    cvcuda_tensor = cvcuda.as_tensor(pycuda_array)
    # docs_tag: end_numpy_pycuda

    # docs_tag: begin_numpy_pycuda_verify
    result_pycuda = gpuarray.GPUArray(
        shape=cvcuda_tensor.shape,
        dtype=cvcuda_tensor.dtype,
        gpudata=cvcuda_tensor.cuda().__cuda_array_interface__["data"][0],
    )
    result_array = result_pycuda.get()

    assert np.allclose(numpy_array, result_array)
    # docs_tag: end_numpy_pycuda_verify


def main():
    """Run all NumPy to GPU transfer examples."""
    numpy_to_cvcuda_via_cuda_python()
    numpy_to_cvcuda_via_torch()
    numpy_to_cvcuda_via_cupy()
    numpy_to_cvcuda_via_pycuda()


if __name__ == "__main__":
    main()
