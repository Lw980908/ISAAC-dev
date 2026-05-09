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
// Create a TensorBatch
#include <nvcv/TensorBatch.hpp>

// Create batch with capacity
nvcv::TensorBatch batch(capacity : 10);

// Create tensors with different shapes (same rank, dtype, layout)
nvcv::Tensor tensor1({100, 100, 3}, nvcv::TYPE_U8, nvcv::TENSOR_HWC);
nvcv::Tensor tensor2({150, 200, 3}, nvcv::TYPE_U8, nvcv::TENSOR_HWC);
nvcv::Tensor tensor3({200, 150, 3}, nvcv::TYPE_U8, nvcv::TENSOR_HWC);

// Add tensors to batch
batch.pushBack(tensor1);
batch.pushBack(tensor2);
batch.pushBack(tensor3);

// Query properties
int32_t rank  = batch.rank();       // Returns 3
int32_t count = batch.numTensors(); // Returns 3
// docs-end: main
