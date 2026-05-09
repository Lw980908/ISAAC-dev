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

import cvcuda


def image_to_tensor() -> None:
    """Convert Image to Tensor with zero-copy wrapping."""
    # Valid conversion: pitch-linear format with no chroma subsampling
    image = cvcuda.Image((640, 480), cvcuda.Format.RGB8)
    tensor = cvcuda.as_tensor(image)  # noqa: F841

    # Valid conversion: single channel format
    grayscale = cvcuda.Image((640, 480), cvcuda.Format.U8)
    tensor_gray = cvcuda.as_tensor(grayscale)  # noqa: F841

    # Invalid conversion: NV12 has chroma subsampling
    try:
        nv12 = cvcuda.Image((1920, 1080), cvcuda.Format.NV12)
        tensor_nv12 = cvcuda.as_tensor(nv12)  # noqa: F841
        raise AssertionError("Expected ValueError for NV12 format")
    except RuntimeError as e:
        assert "sub-sampled" in str(e)


def tensor_to_image() -> None:
    """Convert Tensor to Image using the foreign interface."""
    # Valid conversion: using the foreign interface (.cuda())
    tensor = cvcuda.Tensor((640, 480, 3), cvcuda.Type.U8, layout="HWC")
    image = cvcuda.as_image(tensor.cuda(), format=cvcuda.Format.RGB8)  # noqa: F841

    # Valid conversion: single channel tensor
    tensor_gray = cvcuda.Tensor((640, 480), cvcuda.Type.U8, layout="HW")
    image_gray = cvcuda.as_image(  # noqa: F841
        tensor_gray.cuda(), format=cvcuda.Format.U8
    )

    # Invalid conversion: passing tensor directly without .cuda()
    try:
        tensor_invalid = cvcuda.Tensor((640, 480, 3), cvcuda.Type.U8, layout="HWC")
        image_invalid = cvcuda.as_image(  # noqa: F841
            tensor_invalid, format=cvcuda.Format.RGB8
        )
        raise AssertionError("Expected TypeError when not using foreign interface")
    except TypeError as e:
        assert "ExternalBuffer" in str(e) or "buffer" in str(e).lower()


def main() -> None:
    image_to_tensor()
    tensor_to_image()


if __name__ == "__main__":
    main()
