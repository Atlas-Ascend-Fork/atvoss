/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
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
    using InputsType = typename CurrentOpAssign::RhsType;
    using CurrentInputs = Unique_t<typename ExtractInputs<InputsType>::Type>;

    template <typename InputNode>
    struct IsProcessed {
        static constexpr bool value = false;
    };

    template <size_t N, typename T, ParamUsage U, size_t RN>
    struct IsProcessed<Param<N, T, U, RN>> {
        static constexpr bool value = Contains_v<ProcessedParams, Param<N, T, U, RN>>;
    };

    template <typename InputNode>
    struct NeedInsertCopyIn {
        static constexpr bool value = IsInParam<InputNode>::value && !IsProcessed<InputNode>::value;
    };

    template <typename InputNode>
    struct InsertCopyInIfNeeded {
        using Type = TypeList<>;
    };

    template <size_t N, typename T, ParamUsage U, size_t RN>
    struct InsertCopyInIfNeeded<Param<N, T, U, RN>> {
        using Type = std::conditional_t<
            NeedInsertCopyIn<Param<N, T, U, RN>>::value,
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
    using Output = typename CurrentOpAssign::LhsType;

    using CopyOutOp =
        std::conditional_t<IsOutParam<Output>::value, TypeList<OpAssign<Output, OpCopyOut<Output>>>, TypeList<>>;

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

template <typename ExprList, typename FullList, typename ExcludeList, typename ResultList = TypeList<>>
struct FindLocalVarReferences;

template <typename FullList, typename ExcludeList, typename... ResultTypes>
struct FindLocalVarReferences<TypeList<>, FullList, ExcludeList, TypeList<ResultTypes...>> {
    using Type = TypeList<ResultTypes...>;
    using OtherInputs = TypeList<>;
};

template <typename First, typename... Rest, typename FullList, typename ExcludeList, typename... ResultTypes>
struct FindLocalVarReferences<TypeList<First, Rest...>, FullList, ExcludeList, TypeList<ResultTypes...>> {
public:
    using CurrentResults = TypeList<ResultTypes...>;
    using Output = typename First::LhsType;
    using DirectReferences = typename FilterRefLocalVar<FullList, Output>::Type;
    using NewExclude = Concatenate_t<ExcludeList, TypeList<Output>>;

    using DirectOtherInputs = typename FindAllUnhandledInputs<DirectReferences, FullList, NewExclude>::Type;
    using NewExcludeForRecursive = Concatenate_t<NewExclude, Map_t<ExtractResultType, DirectOtherInputs>>;

    using RecursiveRefResult = FindLocalVarReferences<DirectReferences, FullList, NewExcludeForRecursive>;
    using RecursiveResult = typename RecursiveRefResult::Type;
    using RecursiveOtherInputs = typename RecursiveRefResult::OtherInputs;

    using DirectAndRecursiveOtherInputs = Concatenate_t<DirectOtherInputs, RecursiveOtherInputs>;
    using NewExcludeWithOtherInputs =
        Concatenate_t<NewExcludeForRecursive, Map_t<ExtractResultType, DirectAndRecursiveOtherInputs>>;

    using NextRefResult =
        FindLocalVarReferences<TypeList<Rest...>, FullList, NewExcludeWithOtherInputs, CurrentResults>;
    using NextResult = typename NextRefResult::Type;
    using NextOtherInputs = typename NextRefResult::OtherInputs;

    using CombinedOtherInputs = Concatenate_t<DirectAndRecursiveOtherInputs, NextOtherInputs>;

    using RefsSoFar = Concatenate_t<DirectReferences, RecursiveResult, NextResult>;

    using OtherInputRefs = typename CollectOtherInputRefs<
        CombinedOtherInputs, FullList, NewExcludeWithOtherInputs, Concatenate_t<RefsSoFar, CurrentResults>>::Type;
    using Type = Unique_t<OtherInputRefs>;
    using OtherInputs = Unique_t<CombinedOtherInputs>;
};

template <typename ExprList /*TypeList*/, bool IsBinaryAcc>
struct ReduceAutoDag {
private:
    // Step 1: 识别 Reduce 操作
    using ReduceOpList = Filter_t<ContainsReduceOp, ExprList>;
    using ReduceOpResList = typename Map<ExtractResultType, ReduceOpList>::Type;
    // Step 2: 查找依赖 Reduce 输出的表达式
    using ReduceRefListRaw = typename FindLocalVarReferences<ReduceOpList, ExprList, ReduceOpResList>::Type;
    using ReduceRefList = typename SortExprsByOutput<ReduceRefListRaw>::Type;
    using ReduceRefCopyInList = typename InsertCopyIn<ReduceRefList>::Type;
    using ReducedWithReduceList = Concatenate_t<ReduceOpList, ReduceRefList>;
    using ReducedWithReduceCopyInList = Concatenate_t<ReduceOpList, ReduceRefCopyInList>;
    // Step 3: 划分 PreReduce 和 Reduced
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

public:
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
