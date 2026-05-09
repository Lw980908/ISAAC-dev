# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cvcuda
import pytest as t
import numpy as np


@t.mark.parametrize(
    "type,dt",
    [
        (cvcuda.Type.U8, np.uint8),
        (cvcuda.Type.U8, np.dtype(np.uint8)),
        (cvcuda.Type.S8, np.int8),
        (cvcuda.Type.U16, np.uint16),
        (cvcuda.Type.S16, np.int16),
        (cvcuda.Type.U32, np.uint32),
        (cvcuda.Type.S32, np.int32),
        (cvcuda.Type.U64, np.uint64),
        (cvcuda.Type.S64, np.int64),
        (cvcuda.Type.F32, np.float32),
        (cvcuda.Type.F64, np.float64),
        (cvcuda.Type.C64, np.complex64),
        (cvcuda.Type._2C64, np.dtype("2F")),
        (cvcuda.Type.C128, np.complex128),
        (cvcuda.Type._2C128, np.dtype("2D")),
        (cvcuda.Type._3S8, np.dtype("3i1")),
        (cvcuda.Type._4S32, np.dtype("4i")),
    ],
)
def test_datatype_dtype(type, dt):
    assert type == dt

    t = cvcuda.Type(dt)
    assert t == type
    assert t == dt


@t.mark.parametrize("dt", [np.dtype([("f1", np.uint64), ("f2", np.int32)]), "invalid"])
def test_datatype_dtype_conv_error(dt):
    with t.raises(TypeError):
        cvcuda.Type(dt)
