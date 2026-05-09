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
Common utilities for the samples.

Classes:
    TRT: TensorRT engine class with cvcuda.Tensor inputs and outputs.

Functions:
    get_cache_dir: Get the cache directory for storing models and outputs.
    cuda_memcpy_h2d: Copy host array to device array.
    cuda_memcpy_d2h: Copy device array to host array.
    read_image: Read an image from a file and return a CVCUDA tensor.
    write_image: Write an image to a file.
    export_classifier_onnx: Export a PyTorch classification model to an ONNX model.
    export_retinanet_onnx: Export a PyTorch RetinaNet model to an ONNX model.
    engine_from_onnx: Create a TensorRT engine from an ONNX model.
    parse_image_args: Parse basic arguments for image processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cvcuda
import cuda.bindings.runtime as cudart
import numpy as np
from nvidia import nvimgcodec


def get_cache_dir() -> Path:
    """
    Get the cache directory for storing models, outputs, and temporary files.

    Returns:
        Path to the cvcuda/.cache directory in the CV-CUDA root.
    """
    # Get the CV-CUDA root directory (2 levels up from this file)
    cvcuda_root = Path(__file__).parent.parent
    cache_dir = cvcuda_root / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _cuda_memcpy(
    dst: int,
    src: int,
    size_bytes: int,
    kind: cudart.cudaMemcpyKind,
) -> None:
    """
    Internal: Execute cudaMemcpy with error checking.

    Args:
        dst: Destination pointer
        src: Source pointer
        size_bytes: Number of bytes to copy
        kind: cudaMemcpyKind (HostToDevice, DeviceToHost, etc.)
    """
    (err,) = cudart.cudaMemcpy(dst, src, size_bytes, kind)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy failed: {err}")


def _get_device_ptr(device_array: int | dict | object) -> int:
    """
    Internal: Extract device pointer from various formats.

    Args:
        device_array: Device array (int pointer, __cuda_array_interface__, or dict)

    Returns:
        int: Device pointer address
    """
    if isinstance(device_array, int):
        return device_array
    elif hasattr(device_array, "__cuda_array_interface__"):
        return device_array.__cuda_array_interface__["data"][0]
    elif isinstance(device_array, dict) and "data" in device_array:
        return device_array["data"][0]
    else:
        err_msg = "Invalid device array: "
        err_msg += "must be int pointer, "
        err_msg += "have __cuda_array_interface__, or "
        err_msg += "be a dict with 'data' key"
        raise ValueError(err_msg)


def cuda_memcpy_h2d(
    host_array: np.ndarray,
    device_array: int | dict | object,
) -> None:
    """
    Copy host array to device array.

    Args:
        host_array: Host array to copy.
        device_array: Device array to copy to, __cuda_array_interface__ or object with interface.
    """
    if not host_array.flags.c_contiguous:
        raise ValueError("Host array must be contiguous")

    device_ptr = _get_device_ptr(device_array)
    _cuda_memcpy(
        dst=device_ptr,
        src=host_array.ctypes.data,
        size_bytes=host_array.nbytes,
        kind=cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )


def cuda_memcpy_d2h(
    device_array: int | dict | object,
    host_array: np.ndarray,
) -> None:
    """
    Copy device array to host array.

    Args:
        device_array: Device array to copy from, __cuda_array_interface__ or object with interface.
        host_array: Host array to copy to.
    """
    if not host_array.flags.c_contiguous:
        raise ValueError("Host array must be contiguous")

    device_ptr = _get_device_ptr(device_array)
    _cuda_memcpy(
        dst=host_array.ctypes.data,
        src=device_ptr,
        size_bytes=host_array.nbytes,
        kind=cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )


def zero_copy_split(batch_tensor: cvcuda.Tensor) -> list[cvcuda.Tensor]:
    """
    Split a batch tensor into a list of individual tensors using zero-copy.

    Args:
        batch_tensor: The batch tensor to split.

    Returns:
        A list of individual tensors.
    """
    # Helper object which has a __cuda_array_interface__
    class CudaBuffer:
        __cuda_array_interface__ = None
        obj = None

    # The high-level overview of this function is to create a custom 'view'
    # (using PyTorch terminology) for each individual Tensor in the batch.
    # We will do this by creating a custom CUDA buffer with a __cuda_array_interface__
    # setup from the batch tensor's CUDA buffer.
    # The steps are:
    # 1. Get the CUDA buffer from the batch tensor
    # 2. For each image in the batch, create a custom CUDA buffer with a __cuda_array_interface__
    # setup from the batch tensor's CUDA buffer.
    # 3. Create a new Tensor from the custom CUDA buffer.

    cuda_interface = batch_tensor.cuda().__cuda_array_interface__
    batch_dim = batch_tensor.shape[0]
    height, width, channels = batch_tensor.shape[1:]
    dtype = batch_tensor.dtype

    # Get strides from the tensor to handle padded memory correctly
    strides = cuda_interface.get("strides")
    if strides is None:
        # If no strides, assume C-contiguous layout
        batch_stride_bytes = height * width * channels * dtype.itemsize
        item_strides = None
    else:
        # Use the actual batch stride from the tensor
        batch_stride_bytes = strides[0]
        # Preserve the HWC strides for the resulting tensors
        item_strides = strides[1:]

    new_tensors: list[cvcuda.Tensor] = []
    for i in range(batch_dim):
        offset_ptr = cuda_interface["data"][0] + i * batch_stride_bytes

        offset_buffer = CudaBuffer()
        buffer_interface = {
            "shape": (height, width, channels),
            "typestr": dtype.str,
            "data": (offset_ptr, False),
            "version": 3,
        }

        # Include strides if the tensor has non-contiguous memory layout
        if item_strides is not None:
            buffer_interface["strides"] = item_strides

        offset_buffer.__cuda_array_interface__ = buffer_interface
        offset_buffer.obj = batch_tensor.cuda()

        new_tensors.append(cvcuda.as_tensor(offset_buffer, layout="HWC"))
    return new_tensors


def read_image(
    file: Path,
) -> cvcuda.Tensor:
    """
    Read an image from a file and return a CVCUDA tensor.

    Args:
        file: Path to the image file.

    Returns:
        CVCUDA tensor.
    """
    decoder = nvimgcodec.Decoder()
    nvc_img = decoder.decode(str(file))
    return cvcuda.as_tensor(nvc_img, "HWC")


def write_image(
    tensor: cvcuda.Tensor,
    file: Path | str,
) -> None:
    """
    Write an image to a file.

    Args:
        tensor: CVCUDA tensor to write.
        file: Path to the image file.
    """
    if isinstance(file, str):
        file = Path(file)
    if file.exists():
        file.unlink()

    encoder = nvimgcodec.Encoder()
    nvc_img = nvimgcodec.as_image(tensor.cuda())
    encoder.write(str(file), nvc_img)
    print(f"Saved image to {file}")


def export_classifier_onnx(
    model: "torch.nn.Module",  # noqa: F821
    output: Path,
    input_shape: tuple[int, ...],
    *,
    verbose: bool | None = None,
) -> None:
    """
    Export an ONNX model from a TorchVision classification model.

    Args:
        model: PyTorch model to export.
        output: Path to the output ONNX model.
        input_shape: Shape of the input tensor.
        verbose: Whether to print the verbose output.
    """
    import torch
    import onnx
    import onnxslim

    class ClassifierEnd2End(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            softmax = torch.nn.functional.softmax(self.model(x), dim=1)
            return softmax

    model = ClassifierEnd2End(model)
    model.eval()

    torch.onnx.export(
        model,
        args=(torch.randn(1, *input_shape),),
        f=output,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=verbose,
    )
    onnx_model = onnx.load(output)
    slim_onnx_model = onnxslim.slim(onnx_model)
    onnx.save(slim_onnx_model, output)


def export_segmentation_onnx(
    model: "torch.nn.Module",  # noqa: F821
    output: Path,
    input_shape: tuple[int, ...],
    *,
    verbose: bool | None = None,
) -> None:
    """
    Export an ONNX model from a TorchVision segmentation model.

    Args:
        model: PyTorch model to export.
        output: Path to the output ONNX model.
        input_shape: Shape of the input tensor.
        verbose: Whether to print the verbose output.
    """
    import torch
    import onnx
    import onnxslim

    class SegmentationEnd2End(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            softmax = torch.nn.functional.softmax(self.model(x)["out"], dim=1)
            return softmax

    model = SegmentationEnd2End(model)
    model.eval()

    torch.onnx.export(
        model,
        args=(torch.randn(1, *input_shape),),
        f=output,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=verbose,
    )
    onnx_model = onnx.load(output)
    slim_onnx_model = onnxslim.slim(onnx_model)
    onnx.save(slim_onnx_model, output)


def export_retinanet_onnx(
    model: "torch.nn.Module",  # noqa: F821
    output: Path,
    input_shape: tuple[int, ...],
    num_classes: int = 91,
    max_detections: int = 100,
    *,
    verbose: bool | None = None,
) -> None:
    """
    Export RetinaNet model with backbone + head + EfficientNMS (no complex decoding needed).

    Args:
        model: PyTorch RetinaNet model to export.
        output: Path to the output ONNX model.
        input_shape: Shape of the input tensor.
        num_classes: Number of classes for detection (COCO=91).
        max_detections: Maximum number of detections to keep.
        verbose: Whether to print the verbose output.
    """
    import torch
    import onnx
    import onnxslim

    # Pre-generate anchors for RetinaNet
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        features = model.backbone(dummy_input)
        feat_list = list(features.values())

        # Generate anchors using model's anchor generator
        from torchvision.models.detection.image_list import ImageList

        image_list_obj = ImageList(dummy_input, [(input_shape[1], input_shape[2])])
        anchors_list = model.anchor_generator(image_list_obj, feat_list)
        all_anchors = anchors_list[0]  # [num_anchors, 4] in (x1, y1, x2, y2) format

    # Wrapper to export backbone + head with anchor box decoding
    class BackboneAndHead(torch.nn.Module):
        def __init__(self, model, anchors):
            super().__init__()
            self.backbone = model.backbone
            self.head = model.head
            # Register anchors as buffer
            self.register_buffer("anchors", anchors)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            feat_list = list(features.values())

            # Get predictions from head - RetinaNet returns dict
            head_outputs = self.head(feat_list)
            cls_logits = head_outputs["cls_logits"]  # [B, anchors, num_classes]
            bbox_deltas = head_outputs[
                "bbox_regression"
            ]  # [B, anchors, 4] as [dx, dy, dw, dh]

            # Decode boxes using anchors
            # RetinaNet uses standard encoding: dx, dy, dw, dh
            # Anchors are [x1, y1, x2, y2], convert to [cx, cy, w, h]
            anchor_widths = self.anchors[:, 2] - self.anchors[:, 0]
            anchor_heights = self.anchors[:, 3] - self.anchors[:, 1]
            anchor_ctr_x = self.anchors[:, 0] + 0.5 * anchor_widths
            anchor_ctr_y = self.anchors[:, 1] + 0.5 * anchor_heights

            # Decode: pred_ctr_x = dx * anchor_w + anchor_ctr_x
            dx = bbox_deltas[:, :, 0]
            dy = bbox_deltas[:, :, 1]
            dw = bbox_deltas[:, :, 2]
            dh = bbox_deltas[:, :, 3]

            pred_ctr_x = dx * anchor_widths.unsqueeze(0) + anchor_ctr_x.unsqueeze(0)
            pred_ctr_y = dy * anchor_heights.unsqueeze(0) + anchor_ctr_y.unsqueeze(0)
            pred_w = torch.exp(dw) * anchor_widths.unsqueeze(0)
            pred_h = torch.exp(dh) * anchor_heights.unsqueeze(0)

            # Convert back to [x1, y1, x2, y2]
            pred_boxes = torch.stack(
                [
                    pred_ctr_x - 0.5 * pred_w,
                    pred_ctr_y - 0.5 * pred_h,
                    pred_ctr_x + 0.5 * pred_w,
                    pred_ctr_y + 0.5 * pred_h,
                ],
                dim=-1,
            )

            # Concatenate class and decoded bbox predictions
            output = torch.cat(
                [cls_logits, pred_boxes], dim=-1
            )  # [B, anchors, num_classes+4]

            return output

    wrapped_model = BackboneAndHead(model, all_anchors)
    wrapped_model.eval()

    # Export to ONNX
    temp_output = output.with_suffix(".temp.onnx")
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        torch.onnx.export(
            wrapped_model,
            args=(dummy_input,),
            f=temp_output,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["raw_output"],
            verbose=verbose,
        )

    def _add_efficientnms_to_model(
        model_proto: onnx.ModelProto,
        num_classes: int,
        max_detections: int,
    ) -> None:
        """Add TensorRT EfficientNMS plugin to ONNX model."""
        import onnx
        import onnx.helper as helper

        graph = model_proto.graph

        # The raw_output is [B, anchors, num_classes+4]
        # We need to split it into boxes [B, anchors, 4] and scores [B, anchors, num_classes]

        # Create constants for slicing
        # Slice for classes: [:, :, :num_classes]
        start_cls = helper.make_tensor("start_cls", onnx.TensorProto.INT64, [1], [0])
        end_cls = helper.make_tensor(
            "end_cls", onnx.TensorProto.INT64, [1], [num_classes]
        )
        axes_cls = helper.make_tensor("axes_cls", onnx.TensorProto.INT64, [1], [2])

        # Slice for boxes: [:, :, num_classes:]
        start_box = helper.make_tensor(
            "start_box", onnx.TensorProto.INT64, [1], [num_classes]
        )
        end_box = helper.make_tensor(
            "end_box", onnx.TensorProto.INT64, [1], [num_classes + 4]
        )
        axes_box = helper.make_tensor("axes_box", onnx.TensorProto.INT64, [1], [2])

        # Add constant tensors to graph
        graph.initializer.extend(
            [start_cls, end_cls, axes_cls, start_box, end_box, axes_box]
        )

        # Create slice nodes to split raw_output
        slice_cls = helper.make_node(
            "Slice",
            inputs=["raw_output", "start_cls", "end_cls", "axes_cls"],
            outputs=["class_logits"],
            name="slice_class_logits",
        )

        slice_box = helper.make_node(
            "Slice",
            inputs=["raw_output", "start_box", "end_box", "axes_box"],
            outputs=["box_predictions"],
            name="slice_boxes",
        )

        # Apply sigmoid to class logits to get scores
        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=["class_logits"],
            outputs=["class_scores"],
            name="sigmoid_scores",
        )

        # Create EfficientNMS_TRT plugin node
        nms_node = helper.make_node(
            "EfficientNMS_TRT",
            inputs=["box_predictions", "class_scores"],
            outputs=[
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
            ],
            name="efficient_nms",
            domain="",  # TensorRT plugin domain
            # Plugin attributes
            background_class=-1,
            score_threshold=0.25,
            iou_threshold=0.5,
            max_output_boxes=max_detections,
            score_activation=0,  # 0 = no activation (already applied sigmoid)
            class_agnostic=0,  # 0 = per-class NMS
            box_coding=0,  # 0 = corner encoding (x1,y1,x2,y2)
        )

        # Add nodes to graph
        graph.node.extend([slice_cls, slice_box, sigmoid_node, nms_node])

        # Update graph outputs
        del graph.output[:]
        graph.output.extend(
            [
                helper.make_tensor_value_info(
                    "num_detections", onnx.TensorProto.INT32, [1, 1]
                ),
                helper.make_tensor_value_info(
                    "detection_boxes", onnx.TensorProto.FLOAT, [1, max_detections, 4]
                ),
                helper.make_tensor_value_info(
                    "detection_scores", onnx.TensorProto.FLOAT, [1, max_detections]
                ),
                helper.make_tensor_value_info(
                    "detection_classes", onnx.TensorProto.INT32, [1, max_detections]
                ),
            ]
        )

    # Load ONNX and simplify it
    onnx_model = onnx.load(temp_output)
    slim_onnx_model = onnxslim.slim(onnx_model)
    # Add EfficientNMS plugin
    _add_efficientnms_to_model(slim_onnx_model, num_classes, max_detections)
    onnx.save(slim_onnx_model, output)


def engine_from_onnx(
    onnx: Path,
    output: Path,
    *,
    use_fp16: bool = False,
) -> None:
    """
    Create a TensorRT engine from an ONNX model.

    Args:
        onnx: Path to the ONNX model.
        output: Path to the output TensorRT engine.
        use_fp16: Whether to enable FP16 precision (default: False for compatibility).
    """
    import tensorrt as trt

    trt.init_libnvinfer_plugins(None, "")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    with onnx.open("rb") as model:
        if not parser.parse(model.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError(f"Failed to parse ONNX model. Errors: {errors}")

    config = builder.create_builder_config()

    # Set memory pool limit (8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    # Create or load a timing cache
    timing_cache_path = get_cache_dir() / "timing.cache"
    if timing_cache_path.exists():
        timing_cache = config.create_timing_cache(timing_cache_path.read_bytes())
    else:
        timing_cache = config.create_timing_cache(b"")
    config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Setup flags
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Run the actual build call
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        precision = "FP16" if use_fp16 else "FP32"
        raise RuntimeError(f"Failed to build TensorRT engine with {precision}.")

    # Save the updated timing cache back to the cache directory
    updated_cache = config.get_timing_cache()
    if updated_cache is not None:
        with timing_cache_path.open("wb") as f:
            f.write(updated_cache.serialize())

    with output.open("wb") as f:
        f.write(serialized_engine)


class TRT:
    def __init__(self, engine_path: Path):
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")

        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())

        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine from {engine_path}"
            )

        self._context = self._engine.create_execution_context()

        # Allocate the output tensors (inputs are dynamically setup)
        self._input_names = []
        self._output_names = []
        self._output_tensors = []
        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                self._output_names.append(tensor_name)
                shape = self._engine.get_tensor_shape(tensor_name)

                # Determine dtype based on tensor name
                if "labels" in tensor_name or "num_detections" in tensor_name:
                    dtype = cvcuda.Type.S32
                else:
                    dtype = cvcuda.Type.F32

                # contiguous_tensor = cvcuda.Tensor((np.prod(shape),), dtype=dtype)
                # self._output_tensors.append(contiguous_tensor.reshape(tuple(shape)))
                self._output_tensors.append(cvcuda.Tensor(tuple(shape), dtype))
            else:
                self._input_names.append(tensor_name)

        # Set the output tensor addresses
        for i, tensor in enumerate(self._output_tensors):
            self._context.set_tensor_address(
                self._output_names[i], tensor.cuda().__cuda_array_interface__["data"][0]
            )

    def __call__(self, tensors: list[cvcuda.Tensor]) -> list[cvcuda.Tensor]:
        for i, tensor in enumerate(tensors):
            self._context.set_tensor_address(
                self._input_names[i], tensor.cuda().__cuda_array_interface__["data"][0]
            )

        self._context.execute_async_v3(cvcuda.Stream.current.handle)

        # Synchronize the stream
        cvcuda.Stream.current.sync()

        return self._output_tensors


def parse_image_args(
    output_name: str | None = None,
    output_dir: Path | None = None,
) -> argparse.Namespace:
    """Parse the arguments for the image processing sample."""
    asset_dir = Path(__file__).resolve().parent / "assets"
    image_dir = asset_dir / "images"
    if output_dir is None:
        output_dir = get_cache_dir()

    if output_name is None:
        output_name = "cat.jpg"
    parser = argparse.ArgumentParser(description="Resize an image with CVCUDA.")
    parser.add_argument(
        "--input",
        type=Path,
        default=image_dir / "tabby_tiger_cat.jpg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_dir / output_name,
        help="Path to the output image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Width of the output image.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Height of the output image.",
    )
    return parser.parse_args()


def debug_helper(
    tag: str,
    tensor: cvcuda.Tensor | np.ndarray,
) -> None:
    """
    Prints a bunch of useful info about a tensor.

    Args:
        tag: Tag to print.
        tensor: CVCUDA tensor to debug.
    """
    is_cvcuda = isinstance(tensor, cvcuda.Tensor)
    print(f"{tag} - {'CVCUDA' if is_cvcuda else 'NumPy'}")
    print(f"\tshape: {tensor.shape}")
    print(f"\tdtype: {tensor.dtype}")
    print(f"\tndim: {tensor.ndim}")
    if is_cvcuda:
        print(f"\tlayout: {tensor.layout}")
        cuda = tensor.cuda().__cuda_array_interface__
        print(f"\tstrides: {cuda['strides']}")
    else:
        print(f"\tstrides: {tensor.strides}")
