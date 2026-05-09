// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// docs-start: main
// Create an Image
#include <nvcv/Image.hpp>

// Basic image allocation
nvcv::Image image({640, 480}, nvcv::FMT_RGB8);

// With specific allocator
nvcv::Image image2({1920, 1080}, nvcv::FMT_RGBA8, allocator);

// Multi-planar format (e.g., NV12)
nvcv::Image nv12Image({1920, 1080}, nvcv::FMT_NV12);
// Creates 2 planes: Y (1920x1080) and UV (960x540)

// Wrap external image data
nvcv::ImageDataStridedCuda imageData    = /* ... */;
nvcv::Image                wrappedImage = nvcv::ImageWrapData(imageData);
// docs-end: main
