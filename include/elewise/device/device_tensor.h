/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DEVICE_TENSOR_H
#define DEVICE_TENSOR_H

#include <algorithm>
#include <numeric>
#include "acl/acl.h"
#include "utils/tensor.h"
namespace Atvoss {

template <typename T>
class DeviceTensor {
public:
    DeviceTensor() = default;
    explicit DeviceTensor(Atvoss::Tensor<T>& src)
    {
        ptr_ = src.data();
    }

    T& operator[](std::size_t pos)
    {
        return ptr_[pos];
    }
    const T& operator[](std::size_t pos) const
    {
        return ptr_[pos];
    }

    void Clear()
    {
        if (ptr_ != nullptr) {
            delete[] ptr_;
            ptr_ = nullptr;
        }
    }

    auto GetPtr() const
    {
        return ptr_;
    }
    using IsTensor = void;

private:
    T* ptr_{nullptr};
};
} // namespace Atvoss

#endif