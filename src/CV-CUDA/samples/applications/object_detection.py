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
"""Simple End-to-End object detection example with CVCUDA preprocessing."""

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
    export_retinanet_onnx,
    get_cache_dir,
    parse_image_args,
    read_image,
    write_image,
    cuda_memcpy_d2h,
)


def main() -> None:
    """Object detection with CVCUDA preprocessing and TensorRT+EfficientNMS inference."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_detections.jpg")

    # docs_tag: begin_model_setup
    # 1. Export the ONNX model (RetinaNet backbone + head + EfficientNMS plugin)
    onnx_model_path = get_cache_dir() / f"retinanet_{args.height}x{args.width}.onnx"
    if not onnx_model_path.exists():
        import torchvision  # noqa: E402

        retinanet = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT,
        )
        export_retinanet_onnx(
            retinanet,
            onnx_model_path,
            (3, args.height, args.width),
            verbose=False,
        )

    # 2. Build the TensorRT engine
    trt_model_path = get_cache_dir() / f"retinanet_{args.height}x{args.width}.trtmodel"
    if not trt_model_path.exists():
        engine_from_onnx(onnx_model_path, trt_model_path, use_fp16=False)
    model = TRT(trt_model_path)
    # docs_tag: end_model_setup

    # docs_tag: begin_read_image
    # 3. Read the image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_preprocessing
    # 4. Preprocess the image
    # 4.1 Add a batch dimension
    input_tensor: cvcuda.Tensor = cvcuda.stack([input_image])

    # 4.2 Resize the image
    resized_tensor: cvcuda.Tensor = cvcuda.resize(
        input_tensor, (1, args.height, args.width, 3), cvcuda.Interp.LINEAR
    )

    # 4.3 Convert to float32
    float_tensor: cvcuda.Tensor = cvcuda.convertto(
        resized_tensor, np.float32, scale=1 / 255
    )

    # 4.4 Convert to NCHW layout
    tensor: cvcuda.Tensor = cvcuda.reformat(float_tensor, "NCHW")
    # docs_tag: end_preprocessing

    # docs_tag: begin_inference
    # 5. Run the inference
    input_tensors: list[cvcuda.Tensor] = [tensor]
    output_tensors: list[cvcuda.Tensor] = model(input_tensors)

    # EfficientNMS outputs: [num_detections, boxes, scores, classes]
    num_detections_tensor = output_tensors[0]  # [1, 1] int32
    boxes_tensor = output_tensors[1]  # [1, max_detections, 4] float32
    scores_tensor = output_tensors[2]  # [1, max_detections] float32
    classes_tensor = output_tensors[3]  # [1, max_detections] int32
    # docs_tag: end_inference

    # docs_tag: begin_postprocessing
    # 6. Copy results to host
    num_detections = np.zeros((1, 1), dtype=np.int32)
    boxes = np.zeros((1, 100, 4), dtype=np.float32)
    scores = np.zeros((1, 100), dtype=np.float32)
    classes = np.zeros((1, 100), dtype=np.int32)

    cuda_memcpy_d2h(num_detections_tensor.cuda(), num_detections)
    cuda_memcpy_d2h(boxes_tensor.cuda(), boxes)
    cuda_memcpy_d2h(scores_tensor.cuda(), scores)
    cuda_memcpy_d2h(classes_tensor.cuda(), classes)

    # 7. Draw the detections on the image
    n = num_detections[0, 0]
    orig_h, orig_w = input_image.shape[:2]
    scale_x = orig_w / float(args.width)
    scale_y = orig_h / float(args.height)

    # Create list of bounding boxes
    bboxes: list[cvcuda.BndBoxI] = []
    for idx, box in enumerate(boxes[0]):
        # only assess boxes from the top n detections
        if idx >= n:
            break
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)
        # CVCUDA bbox are (x, y, width, height)
        bbox = (
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        )
        print(f"Box {idx}: {bbox}")

        # create each cvcuda bounding box
        cvcuda_box = cvcuda.BndBoxI(
            box=bbox,
            thickness=2,
            borderColor=(255, 0, 0),
            fillColor=(0, 0, 0, 0),
        )
        bboxes.append(cvcuda_box)

    bndboxes = cvcuda.BndBoxesI(boxes=[bboxes])
    output_image = cvcuda.bndbox(input_image, bndboxes)
    write_image(output_image, args.output)

    # 8. Verify output file exists
    assert args.output.exists()
    # docs_tag: end_postprocessing
    # docs_tag: end_main


if __name__ == "__main__":
    main()
