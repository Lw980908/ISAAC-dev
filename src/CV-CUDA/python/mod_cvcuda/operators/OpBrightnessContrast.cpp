/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpBrightnessContrast.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {

inline Tensor tensorLike(Tensor &src)
{
    const auto &srcShape = src.shape();
    Shape       dstShape = nvcvpy::CreateShape(srcShape);

    return Tensor::Create(dstShape, src.dtype(), src.layout());
}

inline ImageBatchVarShape batchLike(ImageBatchVarShape &src)
{
    ImageBatchVarShape dst = ImageBatchVarShape::Create(src.capacity());
    for (int i = 0; i < src.numImages(); ++i)
    {
        dst.pushBack(Image::Create(src[i].size(), src[i].format()));
    }
    return dst;
}

template<typename Op, typename Src, typename Dst, typename Call>
auto runGuard(Op &op, Src &src, Dst &dst, std::optional<Tensor> &brightness, std::optional<Tensor> &contrast,
              std::optional<Tensor> &brightnessShift, std::optional<Tensor> &contrastCenter,
              std::optional<Stream> &pstream, Call &&call)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {src});
    for (auto &arg : {brightness, contrast, brightnessShift, contrastCenter})
    {
        if (arg)
        {
            guard.add(LockMode::LOCK_MODE_READ, {*arg});
        }
    }
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    call(*pstream, brightness ? *brightness : nvcv::Tensor{nullptr}, contrast ? *contrast : nvcv::Tensor{nullptr},
         brightnessShift ? *brightnessShift : nvcv::Tensor{nullptr},
         contrastCenter ? *contrastCenter : nvcv::Tensor{nullptr});
}

Tensor BrightnessContrastInto(Tensor &dst, Tensor &src, std::optional<Tensor> &brightness,
                              std::optional<Tensor> &contrast, std::optional<Tensor> &brightnessShift,
                              std::optional<Tensor> &contrastCenter, std::optional<Stream> pstream)
{
    auto op = CreateOperator<cvcuda::BrightnessContrast>();
    runGuard(op, src, dst, brightness, contrast, brightnessShift, contrastCenter, pstream,
             [&](Stream &stream, const nvcv::Tensor &brightnessArg, const nvcv::Tensor &contrastArg,
                 const nvcv::Tensor &brightnessShiftArg, const nvcv::Tensor &contrastCenterArg) {
                 op->submit(stream.cudaHandle(), src, dst, brightnessArg, contrastArg, brightnessShiftArg,
                            contrastCenterArg);
             });
    return dst;
}

Tensor BrightnessContrast(Tensor &src, std::optional<Tensor> &brightness, std::optional<Tensor> &contrast,
                          std::optional<Tensor> &brightnessShift, std::optional<Tensor> &contrastCenter,
                          std::optional<Stream> pstream)
{
    auto dst = tensorLike(src);
    return BrightnessContrastInto(dst, src, brightness, contrast, brightnessShift, contrastCenter, pstream);
}

ImageBatchVarShape VarShapeBrightnessContrastInto(ImageBatchVarShape &dst, ImageBatchVarShape &src,
                                                  std::optional<Tensor> &brightness, std::optional<Tensor> &contrast,
                                                  std::optional<Tensor> &brightnessShift,
                                                  std::optional<Tensor> &contrastCenter, std::optional<Stream> pstream)
{
    auto op = CreateOperator<cvcuda::BrightnessContrast>();
    runGuard(op, src, dst, brightness, contrast, brightnessShift, contrastCenter, pstream,
             [&](Stream &stream, const nvcv::Tensor &brightnessArg, const nvcv::Tensor &contrastArg,
                 const nvcv::Tensor &brightnessShiftArg, const nvcv::Tensor &contrastCenterArg) {
                 op->submit(stream.cudaHandle(), src, dst, brightnessArg, contrastArg, brightnessShiftArg,
                            contrastCenterArg);
             });
    return dst;
}

ImageBatchVarShape VarShapeBrightnessContrast(ImageBatchVarShape &src, std::optional<Tensor> &brightness,
                                              std::optional<Tensor> &contrast, std::optional<Tensor> &brightnessShift,
                                              std::optional<Tensor> &contrastCenter, std::optional<Stream> pstream)
{
    auto dst = batchLike(src);
    return VarShapeBrightnessContrastInto(dst, src, brightness, contrast, brightnessShift, contrastCenter, pstream);
}

} // namespace

void ExportOpBrightnessContrast(py::module &m)
{
    using namespace pybind11::literals;
    py::options options;
    options.disable_function_signatures();

    m.def("brightness_contrast", &BrightnessContrast, "src"_a, "brightness"_a = nullptr, "contrast"_a = nullptr,
          "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

	cvcuda.brightness_contrast(src: cvcuda.Tensor, brightness: cvcuda.Tensor, contrast: cvcuda.Tensor, brightness_shift: cvcuda.Tensor, contrast_center: cvcuda.Tensor, stream: Optional[cvcuda.Stream] = None) -> cvcuda.Tensor

        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.

        See also:
            Refer to the CV-CUDA C API reference for the BrightnessContrast operator
            for more details and usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor.
            brightness (cvcuda.Tensor, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain only 1 element. If not specified, the neutral ``0.``
                is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain only 1 element. If not specified, the middle of the
                assumed input type range is used. For floats it is ``0.5``, for unsigned integer
                types it is ``2 * (number_of_bits - 1)``, for signed integer types it is
                ``2 * (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.
    )pbdoc");
    m.def("brightness_contrast_into", &BrightnessContrastInto, "dst"_a, "src"_a, "brightness"_a = nullptr,
          "contrast"_a = nullptr, "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(

	cvcuda.brightness_contrast_into(dst: cvcuda.Tensor, src: cvcuda.Tensor, brightness: cvcuda.Tensor, contrast: cvcuda.Tensor, brightness_shift: cvcuda.Tensor, contrast_center: cvcuda.Tensor, stream: Optional[cvcuda.Stream] = None)

        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.

        See also:
            Refer to the CV-CUDA C API reference for the BrightnessContrast operator
            for more details and usage examples.

        Args:
            src (cvcuda.Tensor): Input tensor.
            dst (cvcuda.Tensor): Output tensor containing the result of the operation.
            brightness (cvcuda.Tensor, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain only 1 element. If not specified, the neutral ``1.``
                is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain only 1 element. If not specified, the neutral ``0.``
                is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain only 1 element. If not specified, the middle of the
                assumed input type range is used. For floats it is ``0.5``, for unsigned integer
                types it is ``2 * (number_of_bits - 1)``, for signed integer types it is
                ``2 * (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None
    )pbdoc");

    // VarShape variants
    m.def("brightness_contrast", &VarShapeBrightnessContrast, "src"_a, "brightness"_a = nullptr, "contrast"_a = nullptr,
          "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(), "stream"_a = nullptr,
          R"pbdoc(

	cvcuda.brightness_contrast(src: cvcuda.ImageBatchVarShape, brightness: cvcuda.Tensor, contrast: cvcuda.Tensor, brightness_shift: cvcuda.Tensor, contrast_center: cvcuda.Tensor, stream: Optional[cvcuda.Stream] = None) -> cvcuda.ImageBatchVarShape

        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.

        The brightness/brightness_shift/contrast/contrast_center tensors' length must match the
        number of samples in the batch.


        See also:
            Refer to the CV-CUDA C API reference for the BrightnessContrast operator
            for more details and usage examples.

        Args:
            src (cvcuda.ImageBatchVarShape): Input tensor.
            brightness (cvcuda.Tensor, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the neutral ``1.`` is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``1.`` is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``0.`` is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain either 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the middle of the assumed input type range is used. For
                floats it is ``0.5``, for unsigned integer types it is
                ``2 * (number_of_bits - 1)``, for signed integer types it is
                ``2 * (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.
    )pbdoc");
    m.def("brightness_contrast_into", &VarShapeBrightnessContrastInto, "dst"_a, "src"_a, "brightness"_a = nullptr,
          "contrast"_a = nullptr, "brightness_shift"_a = nullptr, "contrast_center"_a = nullptr, py::kw_only(),
          "stream"_a = nullptr,
          R"pbdoc(

	cvcuda.brightness_contrast_into(dst: cvcuda.ImageBatchVarShape, src: cvcuda.ImageBatchVarShape, brightness: cvcuda.Tensor, contrast: cvcuda.Tensor, brightness_shift: cvcuda.Tensor, contrast_center: cvcuda.Tensor, stream: Optional[cvcuda.Stream] = None)

        Adjusts the brightness and contrast of the images according to the formula:
        ``out = brightness_shift + brightness * (contrast_center + contrast * (in - contrast_center))``.

        The brightness/brightness_shift/contrast/contrast_center tensors' length must match the
        number of samples in the batch.


        See also:
            Refer to the CV-CUDA C API reference for the BrightnessContrast operator
            for more details and usage examples.

        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            brightness (cvcuda.ImageBatchVarShape, optional): Optional tensor describing brightness multiplier.
                If specified, it must contain 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the neutral ``1.`` is used.
            contrast (cvcuda.Tensor, optional): Optional tensor describing contrast multiplier.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``1.`` is used.
            brightness_shift (cvcuda.Tensor, optional): Optional tensor describing brightness shift.
                If specified, it must contain either 1 or N elements where N is the number of
                input images. If it contains a single element, the same value is used for all
                input images. If not specified, the neutral ``0.`` is used.
            contrast_center (cvcuda.Tensor, optional): Optional tensor describing contrast center.
                If specified, it must contain either 1 or N elements where N is the number of input
                images. If it contains a single element, the same value is used for all input
                images. If not specified, the middle of the assumed input type range is used. For
                floats it is ``0.5``, for unsigned integer types it is
                ``2 * (number_of_bits - 1)``, for signed integer types it is
                ``2 * (number_of_bits - 2)``.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None
    )pbdoc");
}

} // namespace cvcudapy
