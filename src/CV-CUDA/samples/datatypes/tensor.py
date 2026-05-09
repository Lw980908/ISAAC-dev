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

# docs-start: main
import cvcuda
import torch
import numpy as np


def main() -> None:
    # Basic tensor with explicit layout
    tensor1 = cvcuda.Tensor((224, 224, 3), np.uint8, layout="HWC")  # noqa: F841

    # Batch of images
    tensor2 = cvcuda.Tensor((10, 224, 224, 3), np.float32, layout="NHWC")  # noqa: F841

    # For image batch (infers NHWC layout from format)
    tensor3 = cvcuda.Tensor(  # noqa: F841
        nimages=5, imgsize=(640, 480), format=cvcuda.Format.RGB8
    )

    # With row alignment for optimized memory access
    tensor4 = cvcuda.Tensor(  # noqa: F841
        (224, 224, 3), np.uint8, layout="HWC", rowalign=32
    )  # Align rows to 32-byte boundaries

    # Generic N-D tensor
    tensor5 = cvcuda.Tensor((100, 50, 25), np.float32, layout="DHW")  # noqa: F841

    # Wrap existing torch tensor (zero-copy, NHWC)
    torch_tensor = torch.zeros((10, 224, 224, 3), dtype=torch.float32, device="cuda")
    cvcuda_tensor = cvcuda.as_tensor(torch_tensor, layout="NHWC")

    # Common ML layout: NCHW
    torch_nchw = torch.randn((4, 3, 256, 256), dtype=torch.float32, device="cuda")
    cvcuda_nchw = cvcuda.as_tensor(torch_nchw, layout="NCHW")  # noqa: F841

    # Bidirectional: CV-CUDA back to torch (also zero-copy)
    torch_output = torch.as_tensor(cvcuda_tensor.cuda(), device="cuda")  # noqa: F841

    # Video tensor with temporal dimension (Batch, Frames, Height, Width, Channels)
    video_tensor = cvcuda.Tensor(  # noqa: F841
        (2, 30, 720, 1280, 3), np.uint8, layout="NDHWC"
    )


# docs-end: main


if __name__ == "__main__":
    main()
