/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file type_dump.h
 * @brief 编译期类型信息
 */

#ifndef ATVOSS_REDUCE_TYPE_DUMP_H
#define ATVOSS_REDUCE_TYPE_DUMP_H

#include <cstdint>
#include "reduce/utils/reduce_utils.h"

namespace Atvoss::Reduce {

using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

/**
 * @brief 统计 OpAndThen 表达式链中的操作个数
 * @tparam T 表达式类型 (OpAndThen 链或叶子节点)
 */
template <typename T>
struct ExprOpCount {
	static constexpr uint32_t value = 1;
};

template <typename L, typename R>
struct ExprOpCount<OpAndThen<L, R>> {
	static constexpr uint32_t value = 1 + ExprOpCount<R>::value;
};

/**
 * @brief 编译期 DAG 信息 dump: PreReduce/Reduced 阶段的 TypeList 大小及 buf 参数
 * @tparam ReduceDag ReduceAutoDag 类型
 */
template <typename ReduceDag>
struct DumpDagInfo {
	using PreList = typename ReduceDag::PreReduceList;
	using RedList = typename ReduceDag::ReducedList;

#if !defined(__ATVOSS_HOST_ONLY__)
	__aicore__ static void Print()
	{
		AscendC::printf("--- DagInfo ---\n");
		AscendC::printf(" PreReduceList.size    = %d\n", Size_v<PreList>);
		AscendC::printf("   MTE2(CopyIn)        = %d\n", ReduceDag::template GetMTE2Num<true>());
		AscendC::printf("   MTE3(CopyOut)       = %d\n", ReduceDag::template GetMTE3Num<true>());
		AscendC::printf("   TempCalc            = %d\n", ReduceDag::template GetTempBufNum<true>());

		AscendC::printf(" ReducedList.size      = %d\n", Size_v<RedList>);
		AscendC::printf("   MTE2(CopyIn)        = %d\n", ReduceDag::template GetMTE2Num<false>());
		AscendC::printf("   MTE3(CopyOut)       = %d\n", ReduceDag::template GetMTE3Num<false>());
		AscendC::printf("   TempCalc            = %d\n", ReduceDag::template GetTempBufNum<false>());
	}
#endif
};

/**
 * @brief 编译期表达式信息 dump: PreReduceExpr / PostReduceExpr 的操作个数
 * @tparam PreExpr  PreReduceExpr 类型 (OpAndThen 链)
 * @tparam PostExpr PostReduceExpr 类型 (OpAndThen 链)
 */
template <typename PreExpr, typename PostExpr>
struct DumpExprInfo {
#if !defined(__ATVOSS_HOST_ONLY__)
	__aicore__ static void Print()
	{
		AscendC::printf("--- ExprInfo ---\n");
		AscendC::printf(" PreReduceExpr.ops     = %d\n", ExprOpCount<PreExpr>::value);
		AscendC::printf(" PostReduceExpr.ops    = %d\n", ExprOpCount<PostExpr>::value);
	}
#endif
};

/**
 * @brief 统一入口: 同时 dump DAG 和表达式编译期信息
 * @tparam ReduceDag ReduceAutoDag 类型
 * @tparam PreExpr  PreReduceExpr 类型
 * @tparam PostExpr PostReduceExpr 类型
 *
 * 使用方式 (在 __aicore__ 函数中):
 *   DumpReduceTypeInfo<ReduceDag, PreReduceExpr, PostReduceExpr>::Print();
 */
template <typename ReduceDag, typename PreExpr, typename PostExpr>
struct DumpReduceTypeInfo {
#if !defined(__ATVOSS_HOST_ONLY__)
	__aicore__ static void Print()
	{
		AscendC::printf("==== ReduceTypeDump BEGIN ====\n");
		DumpDagInfo<ReduceDag>::Print();
		DumpExprInfo<PreExpr, PostExpr>::Print();
		AscendC::printf("==== ReduceTypeDump END   ====\n");
	}
#endif
};

} // namespace Atvoss::Reduce
#endif // ATVOSS_REDUCE_TYPE_DUMP_H
