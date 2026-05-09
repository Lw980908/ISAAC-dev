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
// Create an ImageBatch (uniform size)
#include <nvcv/ImageBatch.hpp>

// ImageBatch base class is typically used via ImageBatchVarShape
// For uniform-size batches, you can:

// Option 1: Use ImageBatchVarShape with same-sized images
nvcv::ImageBatch uniformBatch(10);
nvcv::Image      img1({640, 480}, nvcv::FMT_RGB8);
nvcv::Image      img2({640, 480}, nvcv::FMT_RGB8);
uniformBatch.pushBack(img1);
uniformBatch.pushBack(img2);

// docs-end: main
