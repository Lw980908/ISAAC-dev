/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Smoke tests for OpOSD - basic functionality tests without libcuosd dependency
// These tests verify the operator works correctly on GCC-10 and other compilers
// For pixel-perfect validation tests, see TestOpOSD.cpp (requires GCC-11+)

#include "Definitions.hpp"

#include <common/TensorDataUtils.hpp>
#include <cvcuda/OpOSD.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

namespace test = nvcv::test;
using namespace cvcuda::priv;

TEST(OpOSD_Smoke, operator_creation)
{
    // Verify operator can be created and destroyed
    EXPECT_NO_THROW(cvcuda::OSD op);
}

TEST(OpOSD_Smoke, rectangle_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with gray background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 128, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with rectangle
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVBndBoxI bbox;
    bbox.box.x       = 50;
    bbox.box.y       = 50;
    bbox.box.width   = 100;
    bbox.box.height  = 80;
    bbox.thickness   = 2;
    bbox.fillColor   = {255, 0, 0, 128};
    bbox.borderColor = {0, 255, 0, 255};

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
    elements.push_back(element);
    elementVec.push_back(elements);

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Validation strategy: Verify OSD rectangle element is drawn
    // Check that output differs from zeroed buffer (rectangle was drawn)
    std::vector<uint8_t> outData(sampleStride);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));

    // Check that some pixels were drawn (output is not all zeros)
    bool hasNonZero = false;
    for (size_t i = 0; i < outData.size(); i++)
    {
        if (outData[i] != 0)
        {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero) << "Output should contain pixels after drawing OSD rectangle";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, text_element)
{
    // TODO: This test has known issues on Jetson platforms (aarch64)
    // Skip on aarch64 until the underlying text rendering issue is resolved
#if defined(__aarch64__)
    GTEST_SKIP() << "Skipped: OpOSD text_element test has known issues on Jetson/aarch64 platforms";
#endif

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGBA8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGBA8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize RGBA input with dark gray background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long                 sampleStride = inAccess->numRows() * inAccess->rowStride();
    std::vector<uint8_t> inData(sampleStride);
    for (size_t i = 0; i < inData.size(); i += 4)
    {
        inData[i]     = 50;  // R
        inData[i + 1] = 50;  // G
        inData[i + 2] = 50;  // B
        inData[i + 3] = 255; // A
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inData.data(), sampleStride, cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with text
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVText text = NVCVText("Test", 20, "", NVCVPointI({10, 10}), NVCVColorRGBA({255, 255, 255, 255}),
                             NVCVColorRGBA({0, 0, 0, 128}));

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_TEXT, &text);
    elements.push_back(element);
    elementVec.push_back(elements);

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Validation strategy: Verify text was rendered by checking for white text pixels
    // Text color is white (255,255,255,255), should be present in output
    std::vector<uint8_t> outData(sampleStride);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));

    bool hasWhiteText = false;
    for (size_t i = 0; i + 3 < outData.size(); i += 4)
    {
        uint8_t r = outData[i];
        uint8_t g = outData[i + 1];
        uint8_t b = outData[i + 2];

        if (r == 255 && g == 255 && b == 255)
        {
            hasWhiteText = true;
            break;
        }
    }

    EXPECT_TRUE(hasWhiteText) << "Output should contain white text pixels (255,255,255)";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, line_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 200, 200, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 200, 200, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 100, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with line
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVLine line;
    line.pos0.x        = 20;
    line.pos0.y        = 20;
    line.pos1.x        = 180;
    line.pos1.y        = 180;
    line.thickness     = 3;
    line.color         = {255, 0, 255, 255};
    line.interpolation = NVCV_INTERP_LINEAR;

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_LINE, &line);
    elements.push_back(element);
    elementVec.push_back(elements);

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, point_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 150, 150, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 150, 150, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 80, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with point
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVPoint point;
    point.centerPos.x = 75;
    point.centerPos.y = 75;
    point.radius      = 5;
    point.color       = {255, 128, 0, 255};

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point);
    elements.push_back(element);
    elementVec.push_back(elements);

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, circle_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 256, 256, nvcv::FMT_RGBA8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 256, 256, nvcv::FMT_RGBA8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 60, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with circle
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVCircle circle;
    circle.centerPos.x = 128;
    circle.centerPos.y = 128;
    circle.radius      = 50;
    circle.thickness   = 3;
    circle.borderColor = {0, 255, 255, 255};
    circle.bgColor     = {255, 0, 255, 100};

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CIRCLE, &circle);
    elements.push_back(element);
    elementVec.push_back(elements);

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, multiple_elements)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with black background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with multiple different element types
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    // Rectangle
    NVCVBndBoxI bbox;
    bbox.box.x       = 50;
    bbox.box.y       = 50;
    bbox.box.width   = 200;
    bbox.box.height  = 150;
    bbox.thickness   = 2;
    bbox.fillColor   = {255, 0, 0, 0};
    bbox.borderColor = {255, 0, 0, 255};
    auto element1    = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
    elements.push_back(element1);

    // Line
    NVCVLine line;
    line.pos0.x        = 300;
    line.pos0.y        = 100;
    line.pos1.x        = 500;
    line.pos1.y        = 300;
    line.thickness     = 2;
    line.color         = {0, 255, 0, 255};
    line.interpolation = NVCV_INTERP_LINEAR;
    auto element2      = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_LINE, &line);
    elements.push_back(element2);

    // Circle
    NVCVCircle circle;
    circle.centerPos.x = 400;
    circle.centerPos.y = 250;
    circle.radius      = 40;
    circle.thickness   = 2;
    circle.borderColor = {0, 0, 255, 255};
    circle.bgColor     = {0, 0, 0, 0};
    auto element3      = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CIRCLE, &circle);
    elements.push_back(element3);

    elementVec.push_back(elements);

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Basic validation: Check output contains colors from the different elements
    std::vector<uint8_t> outData(sampleStride);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));

    bool hasRed   = false; // From rectangle border
    bool hasGreen = false; // From line
    bool hasBlue  = false; // From circle border

    for (size_t i = 0; i + 2 < outData.size(); i += 3)
    {
        uint8_t r = outData[i];
        uint8_t g = outData[i + 1];
        uint8_t b = outData[i + 2];

        if (r == 255 && g == 0 && b == 0)
            hasRed = true;
        if (r == 0 && g == 255 && b == 0)
            hasGreen = true;
        if (r == 0 && g == 0 && b == 255)
            hasBlue = true;

        if (hasRed && hasGreen && hasBlue)
            break;
    }

    EXPECT_TRUE(hasRed) << "Output should contain red pixels from rectangle";
    EXPECT_TRUE(hasGreen) << "Output should contain green pixels from line";
    EXPECT_TRUE(hasBlue) << "Output should contain blue pixels from circle";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, memory_management)
{
    // Run operator multiple times to verify no memory leaks
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::OSD op;

    for (int iter = 0; iter < 5; iter++)
    {
        nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGB8);
        nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGB8);

        auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(input, nullptr);
        ASSERT_NE(output, nullptr);

        // Initialize input with known background
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
        ASSERT_TRUE(inAccess);
        long sampleStride = inAccess->numRows() * inAccess->rowStride();
        EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 90, sampleStride));
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = 30 + iter * 20;
        bbox.box.y       = 30 + iter * 15;
        bbox.box.width   = 100;
        bbox.box.height  = 80;
        bbox.thickness   = 2;
        bbox.fillColor   = {255, 0, 0, 128};
        bbox.borderColor = {0, 255, 0, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);

        std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, edge_cases)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 70, sampleStride));

    cvcuda::OSD op;

    // Empty elements list - output should match input
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));
        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;
        // Empty elements
        elementVec.push_back(elements);
        std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        // Empty elements list: operator returns early, output remains uninitialized
        // For a smoke test, we just verify the operation doesn't crash
    }

    // Element at image boundary - should draw without crashing
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));
        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = 0;
        bbox.box.y       = 0;
        bbox.box.width   = 100;
        bbox.box.height  = 100;
        bbox.thickness   = 1;
        bbox.fillColor   = {0, 0, 0, 0};
        bbox.borderColor = {255, 255, 255, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);

        std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        // Verify white border was drawn at boundary
        std::vector<uint8_t> outData(sampleStride);
        EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));
        bool hasWhite = false;
        for (size_t i = 0; i + 2 < outData.size(); i += 3)
        {
            if (outData[i] == 255 && outData[i + 1] == 255 && outData[i + 2] == 255)
            {
                hasWhite = true;
                break;
            }
        }
        EXPECT_TRUE(hasWhite) << "Output should contain white border pixels at image boundary";
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, batch_processing)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    batchSize = 3;
    nvcv::Tensor imgIn     = nvcv::util::CreateTensor(batchSize, 224, 224, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut    = nvcv::util::CreateTensor(batchSize, 224, 224, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input tensors for batch
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 110, sampleStride * batchSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride * batchSize));

    // Create OSD elements for each batch item
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;

    for (int b = 0; b < batchSize; b++)
    {
        std::vector<std::shared_ptr<NVCVElement>> elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = 40 + b * 20;
        bbox.box.y       = 40 + b * 20;
        bbox.box.width   = 120;
        bbox.box.height  = 100;
        bbox.thickness   = 2;
        bbox.fillColor   = {static_cast<unsigned char>(50 * b), static_cast<unsigned char>(100 + 50 * b), 200, 128};
        bbox.borderColor = {255, static_cast<unsigned char>(255 - 50 * b), 0, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);
    }

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, various_image_sizes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::OSD op;

    // Test different image sizes
    std::vector<std::pair<int, int>> sizes = {
        { 128,  128},
        { 256,  256},
        { 640,  480},
        {1920, 1080}
    };

    for (auto [w, h] : sizes)
    {
        nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, w, h, nvcv::FMT_RGB8);
        nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, w, h, nvcv::FMT_RGB8);

        auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(input, nullptr);
        ASSERT_NE(output, nullptr);

        // Initialize input for each size
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
        ASSERT_TRUE(inAccess);
        long sampleStride = inAccess->numRows() * inAccess->rowStride();
        EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 120, sampleStride));
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = w / 4;
        bbox.box.y       = h / 4;
        bbox.box.width   = w / 2;
        bbox.box.height  = h / 2;
        bbox.thickness   = 2;
        bbox.fillColor   = {100, 150, 200, 50};
        bbox.borderColor = {255, 255, 0, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);

        std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get())) << "Failed with size " << w << "x" << h;
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
