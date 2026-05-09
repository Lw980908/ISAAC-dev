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
"""
Example showcasing how to use CV-CUDA with NvImgCodec.

NvImgCodec images can be used to create CV-CUDA tensors and vice versa.
"""

# docs_tag: begin_imports
from pathlib import Path

import cvcuda

from nvidia import nvimgcodec

# docs_tag: end_imports


def main():
    """NvImgCodec <-> CV-CUDA interoperability example."""

    # 1. Define the decoder and image path
    # docs_tag: begin_init_nvimgcodec
    # setup paths
    img_path = (
        Path(__file__).parent.parent / "assets" / "images" / "tabby_tiger_cat.jpg"
    )
    cvcuda_root = Path(__file__).parent.parent.parent
    output_dir = cvcuda_root / ".cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tabby_tiger_cat_224_224.jpg"

    # create encoder and decoder
    decoder = nvimgcodec.Decoder()
    encoder = nvimgcodec.Encoder()
    # docs_tag: end_init_nvimgcodec

    # 2. NvImgCodec -> CV-CUDA
    # docs_tag: begin_nvimgcodec_to_cvcuda
    nvimgcodec_image = decoder.read(str(img_path))
    cvcuda_tensor = cvcuda.as_tensor(nvimgcodec_image, "HWC")
    # docs_tag: end_nvimgcodec_to_cvcuda

    # 3. CV-CUDA resize
    # docs_tag: begin_cvcuda_resize
    resized_cvcuda_tensor = cvcuda.resize(
        cvcuda_tensor, (224, 224, 3), cvcuda.Interp.LINEAR
    )
    # docs_tag: end_cvcuda_resize

    # 4. CV-CUDA -> NvImgCodec
    # docs_tag: begin_cvcuda_to_nvimgcodec
    new_nvimgcodec_image = nvimgcodec.as_image(resized_cvcuda_tensor.cuda())
    encoder.write(str(output_path), new_nvimgcodec_image)
    # docs_tag: end_cvcuda_to_nvimgcodec


if __name__ == "__main__":
    main()
