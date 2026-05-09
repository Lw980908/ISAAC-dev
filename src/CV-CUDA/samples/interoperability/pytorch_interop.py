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
Example showcasing how to use CV-CUDA with PyTorch.

PyTorch tensors can be used to create CV-CUDA tensors and vice versa.
"""

# docs_tag: begin_imports
import torch
import cvcuda

# docs_tag: end_imports


def main():
    """PyTorch <-> CV-CUDA interoperability example."""

    # 1. PyTorch -> CV-CUDA
    # docs_tag: begin_torch_to_cvcuda
    torch_tensor = torch.randn(10, 10)
    torch_tensor = torch_tensor.cuda()  # move to GPU
    cvcuda_tensor = cvcuda.as_tensor(torch_tensor)
    # docs_tag: end_torch_to_cvcuda

    # 2. CV-CUDA -> PyTorch
    # docs_tag: begin_cvcuda_to_torch
    # Clone so all tensors aren't sharing same GPU buffer
    new_torch_tensor = torch.as_tensor(cvcuda_tensor.cuda())
    cloned_tensor = (
        new_torch_tensor.clone()
    )  # clone so that all tensors don't share same GPU buffer
    assert cloned_tensor.data_ptr() != new_torch_tensor.data_ptr()
    # docs_tag: end_cvcuda_to_torch

    # 3. Ensure tensors are unchanged
    # docs_tag: begin_ensure_tensors_unchanged
    assert torch.equal(torch_tensor, new_torch_tensor)
    # docs_tag: end_ensure_tensors_unchanged


if __name__ == "__main__":
    main()
