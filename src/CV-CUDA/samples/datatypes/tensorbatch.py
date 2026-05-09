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
import numpy as np


def main() -> None:
    # TensorBatch - all tensors must have same rank, dtype, and layout
    batch = cvcuda.TensorBatch(capacity=10)
    tensor1 = cvcuda.Tensor((100, 100, 3), np.uint8, "HWC")
    tensor2 = cvcuda.Tensor((150, 200, 3), np.uint8, "HWC")
    tensor3 = cvcuda.Tensor((200, 150, 3), np.uint8, "HWC")
    batch.pushback([tensor1, tensor2, tensor3])

    # Different datatypes (each batch needs same dtype)
    batch_float = cvcuda.TensorBatch(capacity=5)
    t_float1 = cvcuda.Tensor((100, 100, 3), np.float32, "HWC")
    t_float2 = cvcuda.Tensor((120, 80, 3), np.float32, "HWC")
    batch_float.pushback([t_float1, t_float2])

    batch_int16 = cvcuda.TensorBatch(capacity=5)
    t_int1 = cvcuda.Tensor((50, 50, 1), np.int16, "HWC")
    batch_int16.pushback([t_int1])


# docs-end: main

if __name__ == "__main__":
    main()
