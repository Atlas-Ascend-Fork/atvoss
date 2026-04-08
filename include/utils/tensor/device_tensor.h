/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file device_tensor.h
 * @brief Device端 Tensor 封装模块
 */

#ifndef ATVOSS_DEVICE_TENSOR_H
#define ATVOSS_DEVICE_TENSOR_H

#include <algorithm>
#include <numeric>
#include "acl/acl.h"
#include "utils/tensor/tensor.h"

namespace Atvoss {

/**
 * @brief Device端 Tensor 封装类
 * @tparam T 数据类型（如 float, int32_t 等）
 */
template <typename T>
class DeviceTensor {
public:
    DeviceTensor() = default;

    /**
     * @brief 从 Host Tensor 构造 DeviceTensor
     * @param src Host 端 Tensor 对象引用
     */
    explicit DeviceTensor(Atvoss::Tensor<T>& src)
    {
        ptr_ = src.data();
    }

    /**
     * @brief 元素访问运算符（可修改）
     * @param pos 元素索引位置
     * @return T& 元素引用，可进行读写操作
     */
    T& operator[](std::size_t pos)
    {
        return ptr_[pos];
    }

    /**
     * @brief 元素访问运算符（只读）
     * @param pos 元素索引位置
     * @return const T& 元素常引用，仅可读
     */
    const T& operator[](std::size_t pos) const
    {
        return ptr_[pos];
    }

    /**
     * @brief 清空并释放内存
     */
    void Clear()
    {
        if (ptr_ != nullptr) {
            delete[] ptr_;
            ptr_ = nullptr;
        }
    }

    /**
     * @brief 获取原始数据指针
     * @return T* 数据指针，用于传递给 Kernel
     */
    auto GetPtr() const
    {
        return ptr_;
    }

    /**
     * @brief Tensor 类型标记
     */
    using IsTensor = void;

private:
    /**
     * @brief 数据指针
     */
    T* ptr_{nullptr};
};

} // namespace Atvoss

#endif // ATVOSS_DEVICE_TENSOR_H