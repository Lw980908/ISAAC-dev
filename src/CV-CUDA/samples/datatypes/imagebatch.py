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

# Python only exports VarShape variant, make an alias
ImageBatch = cvcuda.ImageBatchVarShape


def main() -> None:
    # ImageBatch requires all images to have the same dimensions and format
    batch = ImageBatch(capacity=10)
    img1 = cvcuda.Image((640, 480), cvcuda.Format.RGB8)
    img2 = cvcuda.Image((640, 480), cvcuda.Format.RGB8)
    batch.pushback([img1, img2])

    # Different datatypes - but same dimensions
    batch_rgba = ImageBatch(capacity=5)
    img_rgba = cvcuda.Image((640, 480), cvcuda.Format.RGBA8)  # 4 channels
    batch_rgba.pushback([img_rgba])

    batch_float = ImageBatch(capacity=5)
    img_float = cvcuda.Image((640, 480), cvcuda.Format.RGBf32)  # float32 type
    batch_float.pushback([img_float])


# docs-end: main

if __name__ == "__main__":
    main()
