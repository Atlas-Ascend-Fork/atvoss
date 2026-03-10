/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_TILE_EVA_DATA_H
#define ATVOSS_TILE_EVA_DATA_H
#include "evaluator/eval_base.h"
#include "utils/layout/layout.h"
#include "graph/buffer.h"
#include "operators/tile_shape.h"

namespace Atvoss::Tile {

/*!
 * \brief ub to ub copy
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void CopyAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    AscendC::DataCopy(dst, src, operationShape.axis0);
}

/*!
 * \brief Datacopy from src to dst
 * \param[in] src, Input GlobalTensor
 * \param[in] copyCnt, Length of copy data
 * \param[out] dst, Output LocalTensor
 */
template <typename T>
__aicore__ inline void CopyIn(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t copyCnt)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyCnt * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

/*!
 * \brief Datacopy from src to dst
 * \param[in] src, Input LocalTensor
 * \param[in] copyCnt, Length of copy data
 * \param[out] dst, Output GlobalTensor
 */
template <typename T>
__aicore__ inline void CopyOut(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint64_t copyCnt)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyCnt * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

} // namespace Atvoss::Tile

namespace Atvoss::Ele::Tile {

// Partial specialization for CopyIn(E)
template <typename T>
struct Evaluator<OpCopyIn<T>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpCopyIn<T>& op, Context& context) const
    {
        auto& obj = Evaluator<T>{}(op.GetData(), context);
        uint32_t bufferId = Atvoss::Ele::Tile::GetBufferId<typename Context::BuffMaps, T::number, Tile::BufType::PARAM>(
            context.pingPong);
        // AscendC::printf("OpCopyIn, IN[%u]-[%u] context.pingPong: %u\n", T::number, bufferId, context.pingPong);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
        AscendC::Mutex::Lock<PIPE_MTE2>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
        obj.CopyIn(context.gmOffset, context.elementNum);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
        AscendC::Mutex::Unlock<PIPE_MTE2>(bufferId);
        AscendC::Mutex::Lock<PIPE_V>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }
};

// Partial specialization for CopyOut(E)
template <typename T>
struct Evaluator<OpCopyOut<T>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpCopyOut<T>& op, Context& context) const
    {
        uint32_t bufferId = Atvoss::Ele::Tile::GetBufferId<typename Context::BuffMaps, T::number, Tile::BufType::PARAM>(
            context.pingPong);
        // AscendC::printf("OpCopyIn, IN[%u]-[%u] context.pingPong: %u\n", T::number, bufferId, context.pingPong);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
        AscendC::Mutex::Unlock<PIPE_V>(bufferId);
        AscendC::Mutex::Lock<PIPE_MTE3>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
        auto& obj = Evaluator<T>{}(op.GetData(), context);
        obj.CopyOut(context.gmOffset, context.elementNum);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
        AscendC::Mutex::Unlock<PIPE_MTE3>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }
};

// Partial specialization for OpCopy(E)
template <typename T, typename U>
struct Evaluator<OpAssign<T, OpCopy<U>>> {
public:
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpCopy<U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        auto operationShape = GetShape<Operation::Unary>(context.argsTensors);
        // UB to UB copy
        return Atvoss::Tile::CopyAssign(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

// Partial specialization for Alloc(E)
template <typename T>
struct Evaluator<OpAlloc<T>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAlloc<T>& op, Context& context) const
    {
        // T x = 0;
        if constexpr (!std::is_scalar_v<typename T::Type>) {
            auto& obj = Evaluator<T>{}(op.GetData(), context);

            if constexpr (!HasUsage<T>{}) { // tmp param
                uint32_t tmpId =
                    Atvoss::Ele::Tile::GetBufferId<typename Context::BuffMaps, T::number, Tile::BufType::LOCAL_VAR>(
                        context.pingPong);
                // AscendC::printf("OpAlloc, LOCAL_VAR[%u]-[%u] context.pingPong: %u\n", T::number, tmpId,
                // context.pingPong);
                context.bufPools.AllocTensor(obj.GetUbTensor(), tmpId);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
                AscendC::Mutex::Lock<PIPE_V>(tmpId);
#else
                AscendC::PipeBarrier<PIPE_ALL>();
#endif
                return;
            } else if constexpr (
                HasUsage<T>{} && T::usage == Atvoss::ParamUsage::IN ||
                T::usage == Atvoss::ParamUsage::IN_OUT) { // in param: need copy in
                uint32_t inId =
                    Atvoss::Ele::Tile::GetBufferId<typename Context::BuffMaps, T::number, Tile::BufType::PARAM>(
                        context.pingPong);
                context.bufPools.AllocTensor(obj.GetUbTensor(), inId);
                return;
            } else if constexpr (HasUsage<T>{} && T::usage == Atvoss::ParamUsage::OUT) { // out param
                uint32_t outId =
                    Atvoss::Ele::Tile::GetBufferId<typename Context::BuffMaps, T::number, Tile::BufType::PARAM>(
                        context.pingPong);
                context.bufPools.AllocTensor(obj.GetUbTensor(), outId);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
                AscendC::Mutex::Lock<PIPE_V>(outId);
#else
                AscendC::PipeBarrier<PIPE_ALL>();
#endif
                return;
            }
        }
    }
};

// Partial specialization for Free(E)
template <typename T>
struct Evaluator<OpFree<T>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpFree<T>& op, Context& context) const
    {
        auto& obj = Evaluator<T>{}(op.GetData(), context);
        if constexpr (!HasUsage<T>{}) { // tmp param
            uint32_t tmpId =
                Atvoss::Ele::Tile::GetBufferId<typename Context::BuffMaps, T::number, Tile::BufType::LOCAL_VAR>(
                    context.pingPong);
            // AscendC::printf("OpFree, LOCAL_VAR[%u]-[%u] context.pingPong: %u\n", T::number, tmpId, context.pingPong);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
            AscendC::Mutex::Unlock<PIPE_V>(tmpId);
#else
            AscendC::PipeBarrier<PIPE_ALL>();
#endif
            return;
        } else if constexpr (HasUsage<T>{} && T::usage == Atvoss::ParamUsage::IN) { // in param
            uint32_t inId = Atvoss::Ele::Tile::GetBufferId<typename Context::BuffMaps, T::number, Tile::BufType::PARAM>(
                context.pingPong);
            // AscendC::printf("OpFree, IN[%u]-[%u] context.pingPong: %u\n", T::number, inId, context.pingPong);
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
            AscendC::Mutex::Unlock<PIPE_V>(inId);
#else
            AscendC::PipeBarrier<PIPE_ALL>();
#endif
            return;
        }
    }
};

} // namespace Atvoss::Ele::Tile
#endif // ATVOSS_TILE_DATA_H