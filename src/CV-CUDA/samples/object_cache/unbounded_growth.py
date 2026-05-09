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
import random

import cvcuda
import numpy as np


def create_tensor(h: int, w: int) -> None:
    _ = cvcuda.Tensor((h, w, 3), np.float32, cvcuda.TensorLayout.HWC)


def main(iters: int = 10_000) -> None:
    for _ in range(iters):
        h = random.randint(1000, 2000)
        w = random.randint(1000, 2000)
        create_tensor(h, w)


# docs-end: main


if __name__ == "__main__":
    main(iters=100)
