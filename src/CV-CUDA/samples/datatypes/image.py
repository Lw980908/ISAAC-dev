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

from pathlib import Path

# docs-start: main
import cvcuda
import torch
from nvidia import nvimgcodec


def main() -> None:
    # Direct allocation (managed by CV-CUDA)
    img1 = cvcuda.Image((640, 480), cvcuda.Format.RGB8)  # noqa: F841

    # Zero-initialized
    img2 = cvcuda.Image.zeros((640, 480), cvcuda.Format.RGB8)  # noqa: F841

    # Wrapping GPU buffer (zero-copy)
    gpu_buffer = torch.zeros((480, 640, 3), device="cuda", dtype=torch.uint8)
    img3 = cvcuda.as_image(gpu_buffer, format=cvcuda.Format.RGB8)  # noqa: F841

    # Wrapping multiple GPU buffers (for planar formats)
    gpu_channels = [
        torch.zeros((480, 640), device="cuda", dtype=torch.uint8) for _ in range(3)
    ]
    img4 = cvcuda.as_image(gpu_channels, format=cvcuda.Format.RGB8p)  # noqa: F841

    # With row alignment for optimized memory access
    img6 = cvcuda.Image((1920, 1080), cvcuda.Format.RGB8, rowalign=32)  # noqa: F841

    # Loading images from disk with nvimgcodec - Create decoder
    decoder = nvimgcodec.Decoder()

    # Load image from disk
    image_path = (
        Path(__file__).parent / ".." / "assets" / "images" / "tabby_tiger_cat.jpg"
    )
    nic_img = decoder.read(str(image_path))

    # Convert to CV-CUDA Image
    cvcuda_img = cvcuda.as_image(nic_img, format=cvcuda.Format.RGB8)  # noqa: F841


# docs-end: main


if __name__ == "__main__":
    main()
