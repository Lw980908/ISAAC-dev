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
Example showcasing how to use CV-CUDA with PyCUDA.

PyCUDA GPUArrays can be used to create CV-CUDA tensors and vice versa.
"""

# docs_tag: begin_imports
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.gpuarray as gpuarray
import cvcuda

# docs_tag: end_imports


def main():
    """PyCUDA <-> CV-CUDA interoperability example."""

    # 1. PyCUDA -> CV-CUDA
    # docs_tag: begin_pycuda_to_cvcuda
    numpy_array = np.random.randn(10, 10).astype(np.float32)
    pycuda_array = gpuarray.to_gpu(numpy_array)
    cvcuda_tensor = cvcuda.as_tensor(pycuda_array)
    # docs_tag: end_pycuda_to_cvcuda

    # 2. CV-CUDA -> PyCUDA
    # docs_tag: begin_cvcuda_to_pycuda
    new_pycuda_array = gpuarray.GPUArray(
        shape=cvcuda_tensor.shape,
        dtype=cvcuda_tensor.dtype,
        gpudata=cvcuda_tensor.cuda().__cuda_array_interface__["data"][0],
    )
    # docs_tag: end_cvcuda_to_pycuda

    # 3. Ensure tensors are unchanged
    # docs_tag: begin_ensure_tensors_unchanged
    assert np.equal(pycuda_array.get(), new_pycuda_array.get()).all()
    # docs_tag: end_ensure_tensors_unchanged


if __name__ == "__main__":
    main()
