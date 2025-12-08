/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_DEV_TENSOR_H
#define Atvoss_DEV_TENSOR_H

#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

namespace Atvoss {
constexpr size_t MAX_DIMS = 8;

template<typename T>
class Tensor {
public:
    template<size_t N>
    Tensor(T* dataPtr, uint32_t (&inputShape)[N]) : dataPtr_(dataPtr)
    {
        static_assert(N <= MAX_DIMS, "Shape dimension exceeds maximum allowed");
        static_assert(N > 0, "Shape dimension must be greater than 0");
        dims_ = N;
        std::copy(inputShape, inputShape + dims_, shape_);
    }

    template<size_t N>
    Tensor(T* dataPtr, uint32_t (&inputShape)[N], uint32_t dims) : dataPtr_(dataPtr), dims_(dims)
    {
        static_assert(N <= MAX_DIMS, "Shape dimension exceeds maximum allowed");
        static_assert(N > 0, "Shape dimension must be greater than 0");
        std::copy(inputShape, inputShape + dims_, shape_);
    }

    ~Tensor() = default;

    // Get data point
    T* data() const {
        return static_cast<T*>(dataPtr_);
    }

    // Get shape info
    const uint32_t* shape() const {
        return shape_;
    }

    // Get vector corresponding to the shape
    std::vector<uint32_t> shape_vector() const {
        return std::vector<uint32_t>(shape_, shape_ + dims_);
    }

    // Fet dims of shape
    size_t dims() const {
        return dims_;
    }

private:
    uint32_t shape_[MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
    void* dataPtr_ = nullptr;
    size_t dims_ = 0;
};
}
#endif // Atvoss_DEV_TENSOR_H
