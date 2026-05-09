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

import sys


def test_nvcv_types_available_in_cvcuda():
    """Verify nvcv types are accessible through cvcuda module."""
    # Clear any previous imports to test fresh
    for mod in list(sys.modules.keys()):
        if mod.startswith("nvcv") or mod.startswith("cvcuda"):
            del sys.modules[mod]

    import cvcuda

    # Core types from nvcv should be accessible via cvcuda
    assert hasattr(cvcuda, "Tensor"), "cvcuda.Tensor should be available"
    assert hasattr(cvcuda, "as_tensor"), "cvcuda.as_tensor should be available"
    assert hasattr(cvcuda, "Format"), "cvcuda.Format should be available"
    assert hasattr(cvcuda, "Type"), "cvcuda.Type should be available"
    assert hasattr(cvcuda, "ColorSpec"), "cvcuda.ColorSpec should be available"
    assert hasattr(cvcuda, "Image"), "cvcuda.Image should be available"
    assert hasattr(cvcuda, "TensorLayout"), "cvcuda.TensorLayout should be available"

    # cvcuda operators should also be present
    assert hasattr(cvcuda, "resize"), "cvcuda.resize should be available"
    assert hasattr(cvcuda, "gaussian"), "cvcuda.gaussian should be available"


def test_functional_usage():
    """Test that cvcuda types work functionally."""
    # Clear any previous imports to test fresh
    for mod in list(sys.modules.keys()):
        if mod.startswith("nvcv") or mod.startswith("cvcuda"):
            del sys.modules[mod]

    import cvcuda
    import torch

    # Create tensor using cvcuda re-exported types
    torch_tensor = torch.rand(4, 32, 32, 3, dtype=torch.float32, device="cuda")
    nvcv_tensor = cvcuda.as_tensor(torch_tensor, layout="NHWC")

    # Verify it's the right type
    assert isinstance(nvcv_tensor, cvcuda.Tensor)

    # Use cvcuda operator - resize requires full shape for NHWC tensors
    result = cvcuda.resize(nvcv_tensor, (4, 16, 16, 3), cvcuda.Interp.LINEAR)
    assert isinstance(result, cvcuda.Tensor)
    assert result.shape == (4, 16, 16, 3)
