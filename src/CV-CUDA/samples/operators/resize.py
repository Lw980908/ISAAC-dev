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
"""Simple resize example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Resize an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_resized.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_resize
    # 1. Resize the image to the specified width and height
    # Since the image gets read as HWC format,
    # we need to have the same number of dimensions in the output shape
    # i.e. (height, width, 3)
    output_image: cvcuda.Tensor = cvcuda.resize(
        input_image, (args.height, args.width, 3)
    )
    write_image(output_image, args.output)

    # 2. If we have a batch dimension, we can still resize the image
    batched_image: cvcuda.Tensor = input_image.reshape((1, *input_image.shape), "NHWC")
    batched_output_image: cvcuda.Tensor = cvcuda.resize(
        batched_image, (1, args.height, args.width, 3)
    )
    assert batched_output_image.shape == (1, args.height, args.width, 3)
    # docs_tag: end_resize
    # docs_tag: end_main


if __name__ == "__main__":
    main()
