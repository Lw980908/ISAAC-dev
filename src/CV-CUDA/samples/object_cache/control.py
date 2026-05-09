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

# docs-start: limit
import cvcuda


def main() -> None:
    # Get the cache limit (in bytes)
    current_cache_limit = cvcuda.get_cache_limit_inbytes()  # noqa: F841

    # Set the cache limit (in bytes)
    my_new_cache_limit = 12345  # in bytes
    cvcuda.set_cache_limit_inbytes(my_new_cache_limit)


# docs-end: limit


if __name__ == "__main__":
    main()
