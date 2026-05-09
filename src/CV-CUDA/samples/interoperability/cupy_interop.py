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
Example showcasing how to use CV-CUDA with CuPy.

CuPy tensors can be used to create CV-CUDA tensors and vice versa.
"""

# docs_tag: begin_imports
import cupy
import cvcuda

# docs_tag: end_imports


def main():
    """CuPy <-> CV-CUDA interoperability example."""

    # 1. CuPy -> CV-CUDA
    # docs_tag: begin_cupy_to_cvcuda
    cupy_array = cupy.random.random((10, 10), dtype=cupy.float32)
    cvcuda_tensor = cvcuda.as_tensor(cupy_array)
    # docs_tag: end_cupy_to_cvcuda

    # 2. CV-CUDA -> CuPy
    # docs_tag: begin_cvcuda_to_cupy
    new_cupy_array = cupy.asarray(cvcuda_tensor.cuda())
    # docs_tag: end_cvcuda_to_cupy

    # 3. Ensure tensors are unchanged
    # docs_tag: begin_ensure_tensors_unchanged
    assert cupy.all(cupy_array == new_cupy_array)
    # docs_tag: end_ensure_tensors_unchanged


if __name__ == "__main__":
    main()
