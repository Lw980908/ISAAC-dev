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
"""Simple End-to-End semantic segmentation example with CVCUDA preprocessing."""

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
    export_segmentation_onnx,
    get_cache_dir,
    parse_image_args,
    read_image,
    write_image,
    cuda_memcpy_d2h,
    cuda_memcpy_h2d,
    zero_copy_split,
)


def main() -> None:
    """Semantic segmentation with CVCUDA preprocessing and TensorRT inference."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_segmented.jpg")

    # docs_tag: begin_model_setup
    # 1. Download the onnx model (if not already downloaded)
    onnx_model_path = get_cache_dir() / f"fcn_resnet101_{args.height}x{args.width}.onnx"
    if not onnx_model_path.exists():
        import torchvision  # noqa: E402

        fcn_resnet101 = torchvision.models.segmentation.fcn_resnet101(
            weights=torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        )
        export_segmentation_onnx(
            fcn_resnet101, onnx_model_path, (3, args.height, args.width), verbose=False
        )

    # 2. Build the TensorRT engine (if not already built)
    trt_model_path = (
        get_cache_dir() / f"fcn_resnet101_{args.height}x{args.width}.trtmodel"
    )
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
    frame_nhwc: cvcuda.Tensor = cvcuda.stack([input_image])

    # 4.3 Resize the image
    resized_tensor: cvcuda.Tensor = cvcuda.resize(
        frame_nhwc, (1, args.height, args.width, 3), cvcuda.Interp.LINEAR
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
    # 6. Get outputs back to the host
    output: np.ndarray = np.zeros(output_tensor.shape, dtype=output_tensor.dtype)
    cuda_memcpy_d2h(output_tensor.cuda(), output)

    # 7. Postprocess the outputs
    # 7.1 Get the class probabilities for the cat class from 0-255
    # Required to do on CPU, since cvcuda.Tensor doesn't support +,-,*,/ operations
    class_index = 8  # cat (VOC class index)
    # Extract the class probabilities for the given class_index, shape (1, 224, 224)
    class_probs = output[:, class_index : class_index + 1, :, :]  # noqa: E203
    # Move the class dimension to the end to get (1, 224, 224, 1)
    class_probs = np.transpose(class_probs, (0, 2, 3, 1))
    class_probs *= 255.0
    class_probs = class_probs.astype(np.uint8)
    if not class_probs.flags.c_contiguous:
        class_probs = np.ascontiguousarray(class_probs)

    # 7.2 Move the class probabilities to the GPU
    class_probs_tensor = cvcuda.Tensor(class_probs.shape, np.uint8, "NHWC")
    cuda_memcpy_h2d(class_probs, class_probs_tensor.cuda())

    # 7.3 Upscale the masks to match the original image size
    upscaled_masks: cvcuda.Tensor = cvcuda.resize(
        class_probs_tensor,
        (frame_nhwc.shape[0], frame_nhwc.shape[1], frame_nhwc.shape[2], 1),
        cvcuda.Interp.LINEAR,
    )

    # 7.4 Create a blurred background
    # Compute on the smaller resized image to save computation
    blurred_background: cvcuda.Tensor = cvcuda.resize(
        cvcuda.gaussian(
            resized_tensor,
            kernel_size=(15, 15),
            sigma=(5, 5),
            border=cvcuda.Border.REPLICATE,
        ),
        (frame_nhwc.shape[0], frame_nhwc.shape[1], frame_nhwc.shape[2], 3),
        cvcuda.Interp.LINEAR,
    )

    # 7.5 Use joint bilateral filter to create smooth edge on the masks
    gray_nhwc: cvcuda.Tensor = cvcuda.cvtcolor(
        frame_nhwc, cvcuda.ColorConversion.RGB2GRAY
    )
    jb_masks: cvcuda.Tensor = cvcuda.joint_bilateral_filter(
        upscaled_masks,
        gray_nhwc,
        diameter=5,
        sigma_color=50,
        sigma_space=1,
        border=cvcuda.Border.REPLICATE,
    )

    # 7.6 Create an overlay image of the masks
    composite_image: cvcuda.Tensor = cvcuda.composite(
        frame_nhwc,
        blurred_background,
        jb_masks,
        3,
    )

    # 8. Save the overlay image
    hwc_image = zero_copy_split(composite_image)[0]
    write_image(hwc_image, args.output)

    # 9. Verify output file exists
    assert args.output.exists()
    # docs_tag: end_postprocessing
    # docs_tag: end_main


if __name__ == "__main__":
    main()
