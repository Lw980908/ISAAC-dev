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
Example showcasing how to use CV-CUDA with CUDA Python.

CUDA Python memory buffers can be used to create CV-CUDA tensors and vice versa.
"""

# docs_tag: begin_imports
import numpy as np
import cvcuda

from cuda_python_common import CudaBuffer, cuda_memcpy_h2d, cuda_memcpy_d2h

# docs_tag: end_imports


def main():
    """CUDA Python <-> CV-CUDA interoperability example."""

    # 1. CUDA Python -> CV-CUDA
    # docs_tag: begin_cuda_python_to_cvcuda
    # Allocate host data and CUDA buffer
    numpy_array = np.random.randn(10, 10).astype(np.float32)
    cuda_buffer = CudaBuffer(shape=(10, 10), dtype=np.float32)

    # Copy host to GPU
    cuda_memcpy_h2d(
        numpy_array,
        cuda_buffer,
    )

    # Create CV-CUDA tensor from the CUDA buffer
    cvcuda_tensor = cvcuda.as_tensor(cuda_buffer)
    # docs_tag: end_cuda_python_to_cvcuda

    # 2. CV-CUDA -> CUDA Python
    # docs_tag: begin_cvcuda_to_cuda_python
    # Get the CUDA array interface from the CV-CUDA tensor
    # and create a new CUDA buffer from it
    new_cuda_buffer = CudaBuffer.from_cuda(cvcuda_tensor.cuda())
    # docs_tag: end_cvcuda_to_cuda_python

    # 3. Ensure tensors are unchanged
    # docs_tag: begin_ensure_tensors_unchanged
    # Allocate new host buffer
    result_array = np.empty_like(numpy_array)

    # Copy from GPU to host
    cuda_memcpy_d2h(
        new_cuda_buffer,
        result_array,
    )

    # Ensure tensors are unchanged
    assert np.equal(numpy_array, result_array).all()
    # docs_tag: end_ensure_tensors_unchanged


if __name__ == "__main__":
    main()
