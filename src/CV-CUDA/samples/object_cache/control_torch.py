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

# docs-start: default
import cvcuda
import torch


def set_default_cache_limit() -> None:
    # Set the cache limit (in bytes)
    total_mem = torch.cuda.mem_get_info()[1]
    cvcuda.set_cache_limit_inbytes(total_mem // 2)


# docs-end: default


# docs-start: query
def query_cache_size() -> None:
    print(cvcuda.current_cache_size_inbytes())
    img = cvcuda.Image.zeros((1, 1), cvcuda.Format.F32)  # noqa: F841
    print(cvcuda.current_cache_size_inbytes())


# docs-end: query


def main() -> None:
    set_default_cache_limit()
    query_cache_size()


if __name__ == "__main__":
    main()
