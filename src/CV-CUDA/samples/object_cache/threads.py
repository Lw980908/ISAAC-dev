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
import threading

import cvcuda
import numpy as np


def create_tensor_and_clear():
    tensor = cvcuda.Tensor(  # noqa: F841
        (16, 32, 4), np.float32, cvcuda.TensorLayout.HWC
    )
    print(cvcuda.cache_size(), cvcuda.cache_size(cvcuda.ThreadScope.LOCAL))  # 2 1
    cvcuda.clear_cache(cvcuda.ThreadScope.LOCAL)
    print(cvcuda.cache_size(), cvcuda.cache_size(cvcuda.ThreadScope.LOCAL))  # 1 0


def main() -> None:
    tensor = cvcuda.Tensor(  # noqa: F841
        (16, 32, 4), np.float32, cvcuda.TensorLayout.HWC
    )
    thread = threading.Thread(target=create_tensor_and_clear)
    thread.start()
    thread.join()


# docs-end: main

if __name__ == "__main__":
    main()
