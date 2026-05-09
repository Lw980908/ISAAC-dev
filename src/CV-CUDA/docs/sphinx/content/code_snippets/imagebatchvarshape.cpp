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
// Create an ImageBatchVarShape
#include <nvcv/ImageBatch.hpp>

// Create batch with capacity
nvcv::ImageBatchVarShape batch(10);

// Create images with different sizes and formats
nvcv::Image img1({640, 480}, nvcv::FMT_RGB8);
nvcv::Image img2({1280, 720}, nvcv::FMT_RGBA8);
nvcv::Image img3({800, 600}, nvcv::FMT_BGR8);

// Add images to batch
batch.pushBack(img1);
batch.pushBack(img2);
batch.pushBack(img3);

// Query properties
int32_t           count = batch.numImages();    // Returns 3
nvcv::Size2D      maxSz = batch.maxSize();      // Returns {1280, 720}
nvcv::ImageFormat fmt   = batch.uniqueFormat(); // Invalid if formats differ
// docs-end: main
