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


def main() -> None:
    # ImageBatchVarShape - can mix different sizes AND formats
    batch = cvcuda.ImageBatchVarShape(capacity=10)
    img1 = cvcuda.Image((640, 480), cvcuda.Format.RGB8)  # uint8, 3 channels
    img2 = cvcuda.Image((1280, 720), cvcuda.Format.RGBA8)  # uint8, 4 channels
    img3 = cvcuda.Image((800, 600), cvcuda.Format.BGR8)  # uint8, 3 channels
    batch.pushback([img1, img2, img3])

    # Can even mix datatypes in same batch
    batch_mixed = cvcuda.ImageBatchVarShape(capacity=5)
    img_uint8 = cvcuda.Image((640, 480), cvcuda.Format.RGB8)  # uint8
    img_float32 = cvcuda.Image((320, 240), cvcuda.Format.RGBf32)  # float32
    img_gray = cvcuda.Image((800, 600), cvcuda.Format.U8)  # grayscale
    batch_mixed.pushback([img_uint8, img_float32, img_gray])


# docs-end: main

if __name__ == "__main__":
    main()
