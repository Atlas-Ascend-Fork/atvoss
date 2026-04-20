/**
 * Copyright (c) 2026 Huawei Technologies Co.， Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_GRAPH_DAG_H
#define ATVOSS_REDUCE_GRAPH_DAG_H

#include "expression/expr_template.h"
#include "operators/tensor_expression.h"
#include "buffer.h"

namespace Atvoss::Reduce::Graph {

using Atvoss::Util::Contains_v;
using Atvoss::Util::Map;
using Atvoss::Util::Unique_t;

template <typename T>
static constexpr const T& Max(const T& a, const T& b)
{
    return a > b ? a : b;
}

template <typename T>
static constexpr const T& Min(const T& a, const T& b)
{
    return a < b ? a : b;
}

template <
    typename ExprList, typename ProcessedParams = TypeList<>, std::size_t pos = 0, std::size_t Size = Size_v<ExprList>>
struct InsertCopyIn {
private:
    using CurrentOpAssign = Get_t<ExprList, pos>;
    using RHS = typename CurrentOpAssign::RhsType;
    using CurrentInputs = Unique_t<typename ExtractInputs<RHS>::Type>;

    template <typename InputNode>
    struct IsProcessed {
        static constexpr bool value = false;
    };

    template <size_t N, typename T, ParamUsage U, size_t RN>
    struct IsProcessed<Param<N, T, U, RN>> {
        static constexpr bool value = Contains_v<ProcessedParams, Param<N, T, U, RN>>;
    };

    template <typename InputNode>
    struct ShouldInsertCopyIn {
        static constexpr bool value = IsInParam<InputNode>::value && !IsProcessed<InputNode>::value;
    };

    template <typename InputNode>
    struct InsertCopyInIfNeeded {
        using Type = TypeList<>;
    };

    template <size_t N, typename T, ParamUsage U, size_t RN>
    struct InsertCopyInIfNeeded<Param<N, T, U, RN>> {
        using Type = std::conditional_t<
            ShouldInsertCopyIn<Param<N, T, U, RN>>::value,
            TypeList<OpAssign<Param<N, T, U, RN>, OpCopyIn<Param<N, T, U, RN>>>>, TypeList<>>;
    };

    template <typename InputNode>
    struct ProcessInput {
        using Type = TypeList<>;
    };

    template <size_t N, typename T, ParamUsage U, size_t RN>
    struct ProcessInput<Param<N, T, U, RN>> {
        using Type = typename InsertCopyInIfNeeded<Param<N, T, U, RN>>::Type;
    };

    template <typename First, typename... Rest>
    struct ProcessInput<TypeList<First, Rest...>> {
        using FirstProcessed = typename ProcessInput<First>::Type;
        using RestProcessed = typename ProcessInput<TypeList<Rest...>>::Type;
        using Type = Concatenate_t<FirstProcessed, RestProcessed>;
    };

    using InsertedOps = typename ProcessInput<CurrentInputs>::Type;
    using NewProcessedParams = Concatenate_t<ProcessedParams, CurrentInputs>;

public:
    using Type = Concatenate_t<
        InsertedOps, TypeList<CurrentOpAssign>,
        typename InsertCopyIn<ExprList, NewProcessedParams, pos + 1, Size>::Type>;
};

template <typename ExprList, typename ProcessedParams, std::size_t Size>
struct InsertCopyIn<ExprList, ProcessedParams, Size, Size> {
    using Type = TypeList<>;
};

template <typename ExprList, std::size_t pos = 0, std::size_t Size = Size_v<ExprList>>
struct InsertCopyOut {
private:
    using CurrentOpAssign = Get_t<ExprList, pos>;
    using LHS = typename CurrentOpAssign::LhsType;

    using CopyOutOp = std::conditional_t<IsOutParam<LHS>::value, TypeList<OpAssign<LHS, OpCopyOut<LHS>>>, TypeList<>>;

public:
    using Type =
        Concatenate_t<TypeList<CurrentOpAssign>, CopyOutOp, typename InsertCopyOut<ExprList, pos + 1, Size>::Type>;
};

template <typename ExprList, std::size_t Size>
struct InsertCopyOut<ExprList, Size, Size> {
    using Type = TypeList<>;
};

template <typename ExprList>
struct InsertCopyInOut {
    using WithCopyIn = typename InsertCopyIn<ExprList>::Type;
    using Type = typename InsertCopyOut<WithCopyIn>::Type;
};

template <typename ExprList>
struct DagBase {
public:
    // Collect all parameters used (Usage = In/InOut/Out)
    // REMEMBER:
    //  Size_v<AllParams> <= Size_v<InParams> + Size_v<OutParams>
    using AllParams = Params_t<ExprList>;
    // Collect CopyIn parameters (Usage = In/InOut)
    using InParams = Filter_t<IsInVar, AllParams>;
    // Collect CopyOut parameters (Usage = InOut/Out)
    using OutParams = Filter_t<IsOutVar, AllParams>;
};

template <typename ExprList /*TypeList*/, bool IsBinaryAcc>
struct ReduceAutoDag {
private:
    using ReduceOpList = Filter_t<ContainsReduceOp, ExprList>;
    using ReduceOpResList = typename Map<ExtractResultType, ReduceOpList>::Type;
    // 依赖reduce输出的expr
    using ReduceRefList = typename FindLocalVarReferences<ReduceOpList, ExprList, ReduceOpResList>::Type;
    using ReduceRefCopyInList = typename InsertCopyIn<ReduceRefList>::Type;
    using ReducedWithReduceList = Concatenate_t<ReduceOpList, ReduceRefList>;
    using ReducedWithReduceCopyInList = Concatenate_t<ReduceOpList, ReduceRefCopyInList>;

    using PreReduceWithReduceList = Difference_t<ExprList, ReduceRefList>;
    using PreReduceWithReduceCopyInList = typename InsertCopyIn<PreReduceWithReduceList>::Type;
    using PreReduceWoReduceCopyInList = Difference_t<PreReduceWithReduceCopyInList, ReduceOpList>;

public:
    using PreReduceList = typename InsertCopyOut<PreReduceWithReduceCopyInList>::Type;
    using ReducedList = typename InsertCopyOut<ReducedWithReduceCopyInList>::Type;
    using PreReduceBufGen = BufferIdGenerator<PreReduceWoReduceCopyInList, ReduceOpList, IsBinaryAcc>;
    using ReducedBufGen = BufferIdGenerator<ReducedWithReduceCopyInList, ReduceOpList, false>;
    static constexpr auto PreReduceBufIds = PreReduceBufGen::GetBufferIds();
    static constexpr auto ReducedBufIds = ReducedBufGen::GetBufferIds();

private:
    using PreReduceBase = DagBase<PreReduceWithReduceList>;
    using ReducedBase = DagBase<ReducedWithReduceList>;

    using PreReduceAllParams = typename PreReduceBase::AllParams;
    using PreReduceInParams = typename PreReduceBase::InParams;
    using PreReduceOutParams = typename PreReduceBase::OutParams;
    using ReducedAllParams = typename ReducedBase::AllParams;
    using ReducedInParams = typename ReducedBase::InParams;
    using ReducedOutParams = typename ReducedBase::OutParams;

    template <bool IsPre>
    __host_aicore__ constexpr static std::size_t GetMTE2Num()
    {
        if constexpr (IsPre) {
            return Size_v<PreReduceInParams>;
        } else {
            return Size_v<ReducedInParams>;
        }
    }

    template <bool IsPre>
    __host_aicore__ constexpr static std::size_t GetMTE3Num()
    {
        if constexpr (IsPre) {
            return Size_v<PreReduceOutParams>;
        } else {
            return Size_v<ReducedOutParams>;
        }
    }

    template <bool IsPre>
    __host_aicore__ constexpr static int32_t GetTempBufNum()
    {
        if constexpr (IsPre) {
            return PreReduceBufGen::GetTempCalcCount();
        } else {
            return ReducedBufGen::GetTempCalcCount();
        }
    }

    template <bool IsPre>
    __host_aicore__ constexpr static int32_t GetInputMaxDTypeSize()
    {
        if constexpr (IsPre) {
            return PreReduceBufGen::GetInputMaxDTypeSize();
        } else {
            return ReducedBufGen::GetInputMaxDTypeSize();
        }
    }

    template <bool IsPre>
    __host_aicore__ constexpr static int32_t GetInputMinDTypeSize()
    {
        if constexpr (IsPre) {
            return PreReduceBufGen::GetInputMinDTypeSize();
        } else {
            return ReducedBufGen::GetInputMinDTypeSize();
        }
    }

    template <bool IsPre>
    __host_aicore__ constexpr static int32_t GetMaxDTypeSize()
    {
        if constexpr (IsPre) {
            return Max(
                Max(PreReduceBufGen::GetInputMaxDTypeSize(), PreReduceBufGen::GetOutputMaxDTypeSize()),
                PreReduceBufGen::GetLocalVarMaxDTypeSize());
        } else {
            return Max(
                Max(ReducedBufGen::GetInputMaxDTypeSize(), ReducedBufGen::GetOutputMaxDTypeSize()),
                ReducedBufGen::GetLocalVarMaxDTypeSize());
        }
    }

    template <bool IsPre>
    __host_aicore__ constexpr static int32_t GetMinDTypeSize()
    {
        if constexpr (IsPre) {
            return Min(
                Min(PreReduceBufGen::GetInputMinDTypeSize(), PreReduceBufGen::GetOutputMinDTypeSize()),
                PreReduceBufGen::GetLocalVarMinDTypeSize());
        } else {
            return Min(
                Min(ReducedBufGen::GetInputMinDTypeSize(), ReducedBufGen::GetOutputMinDTypeSize()),
                ReducedBufGen::GetLocalVarMinDTypeSize());
        }
    }
};

} // namespace Atvoss::Reduce::Graph
#endif
