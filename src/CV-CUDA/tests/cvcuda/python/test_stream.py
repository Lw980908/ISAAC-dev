# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import cvcuda
import ctypes
import pytest as t
import platform
from packaging import version


def test_stream_gcbag_vs_streamsync_race_condition():
    inputImage = torch.randint(0, 256, (100, 1500, 1500, 3), dtype=torch.uint8).cuda()
    cvcudaInputTensor = cvcuda.as_tensor(inputImage, "NHWC")
    inputmap = torch.randint(0, 256, (100, 1500, 1500, 2), dtype=torch.float).cuda()
    cvcudaInputMap = cvcuda.as_tensor(inputmap, "NHWC")

    cvcuda_stream = cvcuda.Stream()
    with cvcuda_stream:
        cvcudaResizeTensor = cvcuda.remap(cvcudaInputTensor, cvcudaInputMap)
    del cvcudaResizeTensor


def test_current_stream():
    assert cvcuda.Stream.current is cvcuda.Stream.default
    assert type(cvcuda.Stream.current) == cvcuda.Stream


def test_user_stream():
    with cvcuda.Stream():
        assert cvcuda.Stream.current is not cvcuda.Stream.default
    stream = cvcuda.Stream()
    with stream:
        assert stream is cvcuda.Stream.current
        assert stream is not cvcuda.Stream.default
    assert stream is not cvcuda.Stream.default
    assert stream is not cvcuda.Stream.current


def test_nested_streams():
    stream1 = cvcuda.Stream()
    stream2 = cvcuda.Stream()
    assert stream1 is not stream2
    with stream1:
        with stream2:
            assert stream2 is cvcuda.Stream.current
            assert stream1 is not cvcuda.Stream.current
        assert stream2 is not cvcuda.Stream.current
        assert stream1 is cvcuda.Stream.current


def test_wrap_stream_voidp():
    stream = torch.cuda.Stream()

    extStream = ctypes.c_void_p(stream.cuda_stream)

    cvcudaStream = cvcuda.as_stream(extStream)

    assert extStream.value == cvcudaStream.handle


def test_wrap_stream_int():
    stream = torch.cuda.Stream()

    extStream = int(stream.cuda_stream)

    cvcudaStream = cvcuda.as_stream(extStream)

    assert extStream == cvcudaStream.handle


def test_stream_conv_to_int():
    stream = cvcuda.Stream()

    assert stream.handle == int(stream)


class TorchStream:
    def __init__(self, cuda_stream=None):
        if cuda_stream:
            self.m_stream = torch.cuda.ExternalStream(cuda_stream)
        else:
            self.m_stream = torch.cuda.Stream()

    def cuda_stream(self):
        return self.m_stream.cuda_stream

    def stream(self):
        return self.m_stream


@t.mark.parametrize(
    "stream_type",
    [
        TorchStream,
    ],
)
@t.mark.skipif(
    (
        platform.machine() == "aarch64"
        and version.parse(torch.__version__) < version.parse("2.0.0")
    ),
    reason="Test not supported on ARM64 with PyTorch versions < 2.0.0",
)
def test_wrap_stream_external(stream_type):
    extstream = stream_type()

    stream = cvcuda.as_stream(extstream.stream())

    assert extstream.cuda_stream() == stream.handle

    # stream must hold a ref to the external stream, the wrapped cudaStream
    # must not have been deleted
    del extstream

    extstream = stream_type(stream.handle)
    stream = cvcuda.as_stream(extstream.stream())

    assert extstream.cuda_stream() == stream.handle


def test_stream_default_is_zero():
    assert cvcuda.Stream.default.handle == 0


def test_stream_size_in_bytes():
    """
    Checks if the computation of the Stream size in bytes is correct
    """
    stream = cvcuda.Stream()
    assert cvcuda.internal.nbytes_in_cache(stream) == 0
