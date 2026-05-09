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
"""Simple End-to-End classification example with CVCUDA preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import (  # noqa: E402
    TRT,
    engine_from_onnx,
    export_classifier_onnx,
    get_cache_dir,
    parse_image_args,
    read_image,
    cuda_memcpy_d2h,
    cuda_memcpy_h2d,
)


def main() -> None:
    """Classification with CVCUDA preprocessing and TensorRT inference."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args()

    # docs_tag: begin_model_setup
    # 1. Download the onnx model (if not already downloaded)
    onnx_model_path = get_cache_dir() / f"resnet50_{args.height}x{args.width}.onnx"
    if not onnx_model_path.exists():
        import torchvision  # noqa: E402

        resnet50 = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        export_classifier_onnx(
            resnet50, onnx_model_path, (3, args.height, args.width), verbose=False
        )

    # 2. Build the TensorRT engine (if not already built)
    trt_model_path = get_cache_dir() / f"resnet50_{args.height}x{args.width}.trtmodel"
    if not trt_model_path.exists():
        engine_from_onnx(onnx_model_path, trt_model_path)
    model = TRT(trt_model_path)
    # docs_tag: end_model_setup

    # docs_tag: begin_read_image
    # 3. Read the image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_preprocessing
    # 4. Preprocess the image
    # 4.1 Allocate the static imagenet mean and std tensors
    #     This is only needed once and can be reused for all images
    scale: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
        (1, 1, 1, 3)
    )
    scale_tensor: cvcuda.Tensor = cvcuda.Tensor((1, 1, 1, 3), np.float32, "NHWC")
    cuda_memcpy_h2d(scale, scale_tensor.cuda())

    std: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(
        (1, 1, 1, 3)
    )
    std_tensor: cvcuda.Tensor = cvcuda.Tensor((1, 1, 1, 3), np.float32, "NHWC")
    cuda_memcpy_h2d(std, std_tensor.cuda())

    # 4.2 Add a batch dimension
    input_tensor: cvcuda.Tensor = cvcuda.stack([input_image])

    # 4.3 Resize the image
    resized_tensor: cvcuda.Tensor = cvcuda.resize(
        input_tensor, (1, args.height, args.width, 3), cvcuda.Interp.LINEAR
    )

    # 4.4 Convert to float32
    float_tensor: cvcuda.Tensor = cvcuda.convertto(
        resized_tensor, np.float32, scale=1 / 255
    )

    # 4.5 Normalize the image using imagenet mean and std
    normalized_tensor: cvcuda.Tensor = cvcuda.normalize(
        float_tensor,
        scale_tensor,
        std_tensor,
        cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
    )

    # 4.6 Convert to NCHW layout
    tensor: cvcuda.Tensor = cvcuda.reformat(normalized_tensor, "NCHW")
    # docs_tag: end_preprocessing

    # docs_tag: begin_inference
    # 5. Run the inference
    # TRT takes list of tensors and outputs list of tensors
    input_tensors: list[cvcuda.Tensor] = [tensor]
    output_tensors: list[cvcuda.Tensor] = model(input_tensors)
    output_tensor: cvcuda.Tensor = output_tensors[0]
    # docs_tag: end_inference

    # docs_tag: begin_postprocessing
    # 6. Postprocess the inference results
    output: np.ndarray = np.zeros((1, 1000), dtype=np.float32)
    cuda_memcpy_d2h(output_tensor.cuda(), output)

    # 7. Print the top 5 predictions
    indices = np.argsort(output)[0][::-1]
    for i, index in enumerate(indices[:5]):
        print(f"  {i+1}. Class {index}: {output[0][index]}")
    # docs_tag: end_postprocessing
    # docs_tag: end_main


if __name__ == "__main__":
    main()
