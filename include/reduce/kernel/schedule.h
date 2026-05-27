/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file schedule.h
 * \brief Reduce Kernel层调度配置，负责多核切分和Tiling策略计算
 *
 * Kernel层职责：
 * 1. 根据输入Shape和Reduce轴信息，进行Shape变换（消除1维、合并连续轴）
 * 2. 根据Pattern（A/AR/ARA等）计算Tiling策略
 * 3. 计算A轴和R轴的切分参数（SliceDescriptorA_, SliceDescriptorR_）
 * 4. 分配多核任务，计算每个Core处理的Tile数量
 * 5. Kernel执行时，根据BlockIdx计算当前Core需要处理的数据范围
 *
 * Tiling计算流程：
 * 1. CalcBasicBlock - 计算UB可用的BasicBlock大小
 * 2. ComputeCacheLineBlock - 根据CacheLine计算搬运切分
 * 3. ComputeUnitA - 计算A轴切分（保持非Reduce轴）
 * 4. ComputeUnitR - 计算R轴切分（Reduce轴）
 * 5. ComputeProgressUnitA - R轴全载时调整A轴切分
 */

#ifndef ATVOSS_INCLUDE_REDUCE_KERNEL_SCHEDULE_H_
#define ATVOSS_INCLUDE_REDUCE_KERNEL_SCHEDULE_H_
#include <functional>
#include <sstream>
#include <set>
#include <algorithm>
#include "common/arch.h"
#include "common/platform_info.h"
#include "common/string_utils.h"
#include "reduce/utils/reduce_utils.h"
#include "reduce/device/tilingdata.h"
#include "reduce/graph/dag.h"
#include "expression/expr_template.h"
#include "utils/arguments/arguments.h"
#include "utils/math_utils.h"
#include "param_handler.h"

#if !defined(__ATVOSS_HOST_ONLY__)
template <typename T>
using KernelTensor = AscendC::GlobalTensor<T>;
#else
template <typename T>
struct KernelTensor {
    using Type = T;
};
#endif

namespace Atvoss::Reduce {

/*!
 * \brief Reduce Tiling单元，记录单个切分轴的参数
 *
 * Reduce操作需要切分A轴（保持轴）和R轴（Reduce轴），
 * 每个切分单元包含：
 * - idx: 切分发生在哪个轴上
 * - inner: UB内处理的元素数量
 * - outer: 外层循环次数（需要迭代多少次Tile）
 * - step: 每次迭代的步长
 */
struct SliceDescriptor {
    int32_t idx = -1;   // 切分轴索引，-1表示无需切分（全载）
    uint64_t inner = 1; // UB内处理的cacheline数量
    uint64_t outer = 1; // 外层迭代次数
    uint64_t step = 1;  // 每次迭代的步长
    void AssembleUnit(int32_t idx, uint64_t inner, uint64_t outer, uint64_t step)
    {
        idx = idx;
        inner = inner;
        outer = outer;
        step = step;
    }
};

/*!
 * \brief CacheLine切分块，记录CacheLine级别的切分参数
 *
 * 为了保证GM到UB的数据搬运效率，需要按CacheLine大小切分，
 * 减少搬运次数，提升带宽利用率。
 */
struct CacheLineSliceInfo {
    int32_t axis = -1;           // CacheLine切分轴
    uint64_t step = 1;  // CacheLine步长
    uint64_t stepCount = 1; // CacheLine外层迭代次数
    uint64_t aSize = 1;          // CacheLine内A轴大小
    uint64_t rSize = 1;          // CacheLine内R轴大小
};

/*!
 * \brief Reduce Kernel层调度基类
 *
 * 负责Reduce算子的多核切分策略计算和Kernel执行调度。
 *
 * \tparam BlockOp Block层操作类型
 * \tparam Policy Kernel切分策略
 * \tparam ScheduleCfg Kernel配置参数类型
 */
template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class BaseKernelSchedule {
public:
    using ExprMaker = typename BlockOp::ScheduleClz::ExpressMaker; // 表达式构造器
    using ParamStruct = ScheduleCfg;                               // 参数结构体
    using BlockTemplate = BlockOp;                                 // Block模板
    static constexpr auto EXPRESSION = ToLinearizerExpr(ExprMaker{}.template Compute<KernelTensor>());
    using Expr = typename decltype(EXPRESSION)::Type;         // 表达式类型
    using TileShape = typename BlockTemplate::BlockTileShape; // Tile形状
    using ArchTag = typename BlockOp::ScheduleClz::ArchTag;   // 架构标签
    using BlockPolicy = typename BlockOp::BlockPolicy;
    using Params = typename Atvoss::ParamsDeriver<ExprMaker, KernelTensor>::Params;
    using BlockSched = typename BlockOp::ScheduleClz;
    using PreReduceParams = typename BlockSched::PreReduceParams;
    using PostReduceParams = typename BlockSched::PostReduceParams;
    Atvoss::OpPlatformInfo platformInfo_;                // 编译信息

    // ========== 从表达式推导 DAG 信息 ==========
    // 将 OpAndThen 展开为 TypeList，供 ReduceAutoDag 使用
    using ExprList = typename FlattenAtOpAndThen<Expr>::Type;
    // 是否启用二分累加（从 BlockPolicy 获取）
    static constexpr bool EnableBinaryAcc = BlockOp::policy.EnableBinaryAcc;
    // 定义 DAG 类型
    using DagType = Graph::ReduceAutoDag<ExprList, EnableBinaryAcc>;

    // ========== DAG 静态信息（编译期常量） ==========
    // PreReduce 阶段 MTE2 数量（GM → UB 拷入次数）
    static constexpr size_t PreReduceMTE2Num = DagType::template GetMTE2Num<true>();
    // Reduced 阶段 MTE2 数量
    static constexpr size_t ReducedMTE2Num = DagType::template GetMTE2Num<false>();
    // PreReduce 阶段 MTE3 数量（UB → GM 拷出次数）
    static constexpr size_t PreReduceMTE3Num = DagType::template GetMTE3Num<true>();
    // Reduced 阶段 MTE3 数量
    static constexpr size_t ReducedMTE3Num = DagType::template GetMTE3Num<false>();
    // PreReduce 临时 buffer 数量
    static constexpr int32_t PreReduceTempBufNum = DagType::template GetTempBufNum<true>();
    // Reduced 临时 buffer 数量
    static constexpr int32_t ReducedTempBufNum = DagType::template GetTempBufNum<false>();
    // 输入最大 dtype 字节数
    static constexpr int32_t MaxInputDTypeSize = DagType::template GetInputMaxDTypeSize<true>();
    // 输入最小 dtype 字节数
    static constexpr int32_t MinInputDTypeSize = DagType::template GetInputMinDTypeSize<true>();
    // 最大 dtype 字节数（输入、输出、临时变量的最大值）
    static constexpr int32_t MaxDTypeSize = DagType::template GetMaxDTypeSize<true>();
    // 最小 dtype 字节数
    static constexpr int32_t MinDTypeSize = DagType::template GetMinDTypeSize<true>();


    // ========== 常量配置 ==========
    static constexpr uint64_t TILEA = TileShape::template get_type<0>::value; // 用户配置的A轴Tile大小
    static constexpr uint64_t TILER = TileShape::template get_type<1>::value; // 用户配置的R轴Tile大小
    static constexpr int32_t MAX_INNER_A = 512;                               // A轴UB内最大512字节
    static constexpr int32_t MEMBOUND_CORENUM = 40;                           // MemoryBound场景核心数阈值
    static constexpr int32_t VECBOUND_CORENUM = 48;                           // VectorBound场景核心数阈值
    static constexpr int32_t THRES_CORENUM = MEMBOUND_CORENUM;                // 分核阈值
    static constexpr int32_t AXIS_ITERATION_STRIDE = 2;                       // 轴间隔（A/R交替）
    static constexpr uint64_t CACHE_BUF_SIZE = 16 * 1024UL;                   // Binary Reduce缓存
    static constexpr uint64_t vRegSize = 256;                                 // Vector寄存器大小
    static constexpr double maxCoreRatio = 0.95;                              // 分核数和最大分核数的比例
    static constexpr auto& memPolicy = BlockOp::memPolicy;

    // ========== Shape变换相关成员 ==========
    uint64_t shape_[MAX_DIM] = {0};     // 变换后的Shape
    int32_t newShapeSize_ = 1;
    uint64_t srcStride_[MAX_DIM] = {0};    // 输入Stride
    uint64_t outStride_[MAX_DIM] = {0};    // 输出Stride
    int32_t useNddma_ = 0;
    int32_t inputDtypeSize_ = 1;
    uint64_t shapeA_ = 1;
    uint64_t shapeR_ = 1;
    uint64_t reservedNode_ = 0;
    uint64_t reservedSize_ = 0;

    // ========== Tiling计算结果 ==========
    uint64_t basicBlock_;  // UB搬入的buffer大小
    uint64_t resultBlock_; // Reduce计算后的buffer大小
    SliceDescriptor SliceDescriptorA_;  // A轴切分单元
    SliceDescriptor SliceDescriptorR_;  // R轴切分单元
    CacheLineSliceInfo cBlock_;   // CacheLine切分块

    /*!
     * \brief 计算Kernel层调度配置
     *
     * 从用户输入参数中提取Shape和Reduce轴信息，
     * 执行Shape变换和Tiling计算，生成ReduceTilingData。
     *
     * \param[in] arguments 用户输入参数（包含Tensor和axes属性）
     * \param[out] tilingData 统一的Tiling数据
     * \return true 成功，false 失败
     */
    template <typename Args>
    bool MakeScheduleConfig(const Args& arguments, ReduceTilingData& tilingData)
    {
        //先注掉，目前从GetOpPlatformInfo()拿不到对应芯片数据。
        // platformInfo_ = Atvoss::GetOpPlatformInfo();
        std::vector<uint64_t> shapeInfo;
        std::vector<int> axes;

        if (!ValidateArguments(arguments, shapeInfo, axes)) {
            return false;
        }

        if (!ComputeAndSetTilingData(tilingData, shapeInfo, axes)) {
            return false;
        }

        return true;
    }
        // inputDtypeSize_ = 

private:
    /*!
     * \brief 校验输入参数
     *
     * 从arguments中提取tensor和attr信息，校验shape和axes的正确性。
     *
     * \param[in] arguments 用户输入参数
     * \return true 校验通过，false 校验失败
     */
    template <typename Args>
    bool ValidateArguments(const Args& arguments, std::vector<uint64_t>& shapeInfo, std::vector<int>& axes)
    {
        auto& tensorTuple = std::get<0>(arguments);
        auto& attrTuple = std::get<1>(arguments);

        std::vector<uint64_t> inputShape;
        std::vector<uint64_t> outputShape;

        if (!ValidateShape(tensorTuple, inputShape, outputShape, shapeInfo)) {
            return false;
        }

        if (!ValidateAxes(attrTuple, inputShape, axes)) {
            return false;
        }

        if (!ValidateReduceOutputShape(inputShape, axes, outputShape)) {
            return false;
        }
        return true;
    }

    /*!
     * \brief 校验输入输出Shape
     *
     * 从tensor tuple中提取所有Tensor的shape，校验输入输出shape的正确性。
     *
     * \param[in] tensorTuple Tensor元组
     * \param[out] inputShape reduce前的输入shape
     * \param[out] outputShape reduce后的输出shape
     * \return true 校验通过，false 校验失败
     */
    template <typename TupleType>
    bool ValidateShape(
        const TupleType& tensorTuple, std::vector<uint64_t>& inputShape, std::vector<uint64_t>& outputShape, std::vector<uint64_t>& shapeInfo)
    {
        if (!ValidateInputOutputShapes(tensorTuple, inputShape, outputShape)) {
            return false;
        }

        shapeInfo = inputShape;
        if (shapeInfo.size() == 0) {
            printf("[ERROR]: [Atvoss][Tiling] Shape info error \n");
            return false;
        }

        printf("[INFO]: [Atvoss][Tiling] Shape validation passed\n");
        return true;
    }

    /*!
     * \brief 校验Reduce轴参数
     *
     * 从attr tuple中提取axes和keep_dim属性，校验axes的正确性，
     * 并验证reduce后的output shape是否符合预期。
     *
     * \param[in] attrTuple 属性元组
     * \param[in] inputShape reduce前的输入shape
     * \param[out] keepDim 是否保留reduce轴维度
     * \return true 校验通过，false 校验失败
     */
    template <typename AttrTupleType>
    bool ValidateAxes(
        const AttrTupleType& attrTuple, const std::vector<uint64_t>& shapeInfo, std::vector<int>& axes)
    {
        auto* axesPtr = FindAttrByName(attrTuple, "axis");
        if (axesPtr == nullptr) {
            printf("[ERROR]: [Atvoss][Tiling] Cannot find 'axes' attribute\n");
            return false;
        }
        axes = *axesPtr;

        MakeWrapDim(axes, shapeInfo);
        if (!CheckAxes(axes, shapeInfo)) {
            return false;
        }
        printf("[INFO]: [Atvoss][Tiling] Axes validation passed\n");
        return true;
    }

    /*!
     * \brief 执行Tiling计算
     *
     * 根据shape和axes信息，执行Shape变换和Tiling策略计算。
     *
     * \return true 计算成功，false 计算失败
     */
    bool ComputeAndSetTilingData(ReduceTilingData& tilingData, const std::vector<uint64_t>& shapeInfo, std::vector<int>& axes)
    {
        TransformShape(shapeInfo, axes);

        if(!DoTilingMatchPattern(tilingData)){
            return false;
        }
        printf("[INFO]: [Atvoss][Tiling] Tiling computation passed\n");
        return true;
    }

    template <class Pattern>
    void SetTilingData(ReduceTilingData& tilingData)
    {
        int8_t aCount = 0;
        int8_t rCount = 0;
        int8_t innerACount = 0;
        int8_t innerRCount = 0;
        uint64_t perCoreNum = CeilDiv(SliceDescriptorA_.outer * SliceDescriptorR_.outer, platformInfo_.vectorCoreNum);
        uint64_t numBlocks = CeilDiv(SliceDescriptorA_.outer * SliceDescriptorR_.outer, perCoreNum);
        if (SliceDescriptorA_.outer < numBlocks) {
            auto tmpBlockDim = CeilAlign(numBlocks, SliceDescriptorA_.outer);
            if (tmpBlockDim <= platformInfo_.vectorCoreNum) {
                numBlocks = tmpBlockDim;
            } else {
                numBlocks = FloorAlign(numBlocks, SliceDescriptorA_.outer);
            }
        }

        tilingData.ubFactorA = SliceDescriptorA_.step;
        uint64_t tileACntPerCore = CeilDiv(SliceDescriptorA_.outer, numBlocks);
        tilingData.tileACntPerCore = tileACntPerCore;
        tilingData.totalTileACnt = SliceDescriptorA_.outer;

        tilingData.ubFactorR = SliceDescriptorR_.step;
        uint64_t tileRCntPerCore = CeilDiv(SliceDescriptorR_.outer, CeilDiv(numBlocks, SliceDescriptorA_.outer));
        tilingData.tileRCntPerCore = tileRCntPerCore;
        tilingData.totalTileRCnt = SliceDescriptorR_.outer;
        int16_t groupR = CeilDiv(SliceDescriptorR_.outer, tileRCntPerCore);
        tilingData.groupR = groupR;
        if (groupR == 1UL) {
            // normal case
            aCount = (SliceDescriptorA_.idx - (Pattern::FirstA ? 0 : 1)) / AXIS_ITERATION_STRIDE + 1;
            innerRCount = (SliceDescriptorR_.idx - (Pattern::FirstA ? 1 : 0)) / AXIS_ITERATION_STRIDE + 1;
        } else {
            // group case
            aCount = (SliceDescriptorA_.idx - (Pattern::FirstA ? 0 : 1)) / AXIS_ITERATION_STRIDE + 1;
            rCount = (SliceDescriptorR_.idx - (Pattern::FirstA ? 1 : 0)) / AXIS_ITERATION_STRIDE + 1;
            rCount = rCount + aCount;
        }
        tilingData.loopACount = aCount;
        tilingData.loopRCount = rCount;
        for (int32_t i = 0; i < MAX_DIM; i++) {
            tilingData.shape[i] = shape_[i];
            tilingData.inStride[i] = srcStride_[i];
            tilingData.outStride[i] = outStride_[i];
        }
        uint32_t realCore = CeilDiv(SliceDescriptorA_.outer, tileACntPerCore) * CeilDiv(SliceDescriptorR_.outer, tileRCntPerCore);
        tilingData.coreNum = realCore;
        tilingData.useNddma = useNddma_;
        tilingData.resultBlock = resultBlock_;
        tilingData.basicBlock = basicBlock_;
    }
    /*!
     * \brief 计算Shape总元素数
     */
    void MakeWrapDim(std::vector<int>& axes, const std::vector<uint64_t>& shapeInfo)
    {
        size_t shapeSize = shapeInfo.size();
        for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] < 0) {
                axes[i] += shapeSize;
            }
        }
        std::sort(axes.begin(), axes.end());
    }

    bool CheckAxes(const std::vector<int>& axes, const std::vector<uint64_t>& shapeInfo)
    {
        int64_t shapeSize = static_cast<int64_t>(shapeInfo.size());
        int64_t axesSize = static_cast<int64_t>(axes.size());
        if (axesSize == 0) {
            printf("[ERROR]: [Atvoss][Tiling] illegal axes size:%ld", axesSize);
            return false;
        }
        if (axesSize > shapeSize) {
            printf("[ERROR]: [Atvoss][Tiling] illegal axes size:%ld over shape size:%ld", axesSize, shapeSize);
            return false;
        }
        for (int64_t i = 0; i < axesSize; i++) {
            if (axes[i] >= shapeSize || axes[i] < 0) {
                printf("[ERROR]: [Atvoss][Tiling] illegal axis:%ld dim:%ld out of shape range:[0, %ld)", i, axes[i], shapeSize);
                return false;
            }
        }
        return true;
    }
    /*!
     * \brief 根据Shape维度数选择对应的Pattern进行Tiling计算
     *
     * 支持的Pattern：
     * - 1维: A (纯A轴)
     * - 2维: AR (A轴+R轴)
     * - 3维: ARA
     * - 4维: ARAR
     * - 5-9维: 扩展Pattern
     */

    //增加bool返回值
    bool DoTilingMatchPattern(ReduceTilingData& tilingData)
    {
        switch (newShapeSize_) {
            case CONST1:
                return ComputeTiling<A>(tilingData);
            case CONST2:
                return ComputeTiling<AR>(tilingData);
            case CONST3:
                return ComputeTiling<ARA>(tilingData);
            case CONST4:
                return ComputeTiling<ARAR>(tilingData);
            case CONST5:
                PadDimOne<ARARA>();
                return ComputeTiling<ARARARARA>(tilingData);
            case CONST6:
                PadDimOne<ARARAR>();
                return ComputeTiling<ARARARAR>(tilingData);
            case CONST7:
                PadDimOne<ARARARA>();
                return ComputeTiling<ARARARARA>(tilingData);
            case CONST8:
                return ComputeTiling<ARARARAR>(tilingData);
            case CONST9:
                return ComputeTiling<ARARARARA>(tilingData);
            default:
                printf("[ERROR]: [Atvoss][Tiling] Unsupported pattern\n");
                return false;
        }
    }

    template <class Pattern>
    void PadDimOne()
    {
        int32_t padNum = Pattern::TailA ? CONST9 - Pattern::Dim : CONST8 - Pattern::Dim;
        int32_t maxDim = Pattern::TailA ? CONST9 : CONST8;
        if (padNum == 0) {
            return;
        }
        for (int32_t i = 0; i < Pattern::Dim; ++i) {
            shape_[maxDim - 1 - i] = shape_[Pattern::Dim - 1 - i];
        }
        for (int32_t i = 0; i < padNum; ++i) {
            shape_[i] = 1UL;
        }
    }

    /*!
     * \brief 处理空Tensor场景的Tiling计算
     *
     * 当某个Reduce轴为0时，输出为空或全0，需要特殊处理。
     */
    template <class Pattern>
    void ComputeEmptyTiling(ReduceTilingData& tilingData)
    {
        uint64_t outSize = 1;
        for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
            if (IsAxisA<Pattern>(dim)) {
                outSize *= shape_[dim];
            }
        }
        if (outSize == 0) {
            return;
        }
        uint64_t ubAvilSize = platformInfo_.ubSize - CACHE_BUF_SIZE;
        basicBlock_ = FloorAlign(ubAvilSize / CONST2, vRegSize); // double buffer
        for (int32_t i = 1; i < MAX_DIM; i++) {
            shape_[i] = 0;
        }
        shape_[0] = outSize;
        // 空tensor去除R轴后，作为全A的pattern计算切分
        ComputeCacheLineBlock<A>();
        SliceDescriptorA_.outer *= cBlock_.stepCount;
        ComputeUnitA<A>();
        SetTilingData<A>(tilingData);
    }

    /*!
     * \brief 核心Tiling计算函数，按5步流程计算切分参数
     *
     * Tiling计算流程：
     * 1. CalcBasicBlock - 根据计算图计算BasicBlock大小
     * 2. ComputeCacheLineBlockAndUnit - CacheLine切分，保证搬运效率
     * 3. ComputeUnitA - 计算A轴切分，最大512B或分核效率85%
     * 4. ComputeUnitR - 计算R轴切分，满BasicBlock
     * 5. ComputeProgressUnitA - R轴全载时调整A轴切分
     */
    template <class Pattern>
    bool ComputeTiling(ReduceTilingData& tilingData)
    {
        printf("[INFO]: [Atvoss][Tiling] ========== ComputeTiling Start ==========\n");
        printf("[INFO]: [Atvoss][Tiling] Pattern ID: %u, Dim: %d\n", Pattern::ID, Pattern::Dim);
        printf(
            "[INFO]: [Atvoss][Tiling] MemPolicy: %s\n", memPolicy == Atvoss::MemMngPolicy::MANUAL ? "MANUAL" : "AUTO");

        ComputeStride<Pattern>();

        if (IsEmtpyTensor<Pattern>()) {
            printf("[INFO]: [Atvoss][Tiling] Empty tensor detected\n");
            return ComputeEmptyTilingPath<Pattern>(tilingData);
        }

        CalcBasicBlock<Pattern>();

        if (memPolicy == Atvoss::MemMngPolicy::MANUAL) {
            return ComputeManualTilingPath<Pattern>(tilingData);
        } else {
            return ComputeAutoTilingPath<Pattern>(tilingData);
        }
    }

    template <class Pattern>
    bool ComputeEmptyTilingPath(ReduceTilingData& tilingData)
    {
        if (memPolicy == Atvoss::MemMngPolicy::MANUAL) {
            printf("[INFO]: [Atvoss][Tiling] Empty tensor with MANUAL policy\n");
            bool result = CheckAndCalEmptyTiling<Pattern>(TILEA, TILER);
            if (result) {
                printf("[INFO]: [Atvoss][Tiling] Empty tensor tiling check passed\n");
                SetTilingData<Pattern>(tilingData);
            } else {
                printf("[ERROR]: [Atvoss][Tiling] Empty tensor tiling check failed\n");
            }
            return result;
        } else {
            printf("[INFO]: [Atvoss][Tiling] Empty tensor with AUTO policy\n");
            ComputeEmptyTiling<Pattern>(tilingData);
            printf("[INFO]: [Atvoss][Tiling] Empty tensor tiling computed successfully\n");
            return true;
        }
    }

    template <class Pattern>
    bool ComputeManualTilingPath(ReduceTilingData& tilingData)
    {
        printf("[INFO]: [Atvoss][Tiling] MANUAL policy path\n");

        if (!CheckTileShape<Pattern>(TILEA, TILER)) {
            printf("[ERROR]: [Atvoss][Tiling] Tile shape check failed\n");
            return false;
        }

        SetSliceDescriptorWithTileShapeA<Pattern>(TILEA);
        if (Pattern::ID != PATTERN_A) {
            SetSliceDescriptorWithTileShapeR<Pattern>(TILER);
        }
        SetTilingData<Pattern>(tilingData);
        PrintTilingResult();
        printf("[INFO]: [Atvoss][Tiling] ========== ComputeTiling End ==========\n");
        return true;
    }

    template <class Pattern>
    bool ComputeAutoTilingPath(ReduceTilingData& tilingData)
    {
        printf("[INFO]: [Atvoss][Tiling] AUTO policy path\n");
        ComputeCacheLineBlockAndUnit<Pattern>();

        ComputeUnitA<Pattern>();

        ComputeUnitR<Pattern>();

        ComputeProgressUnitA<Pattern>();

        useNddma_ = IsUseNddma<Pattern>();

        SetTilingData<Pattern>(tilingData);
        PrintTilingResult();
        printf("[INFO]: [Atvoss][Tiling] ========== ComputeTiling End ==========\n");
        return true;
    }

    void PrintTilingResult()
    {
        printf("[INFO]: [Atvoss][Tiling] Tiling Result:\n");
        printf(
            "[INFO]: [Atvoss][Tiling]   A-axis: outer=%lu, inner=%lu, step=%lu, idx=%d\n", SliceDescriptorA_.outer,
            SliceDescriptorA_.inner, SliceDescriptorA_.step, SliceDescriptorA_.idx);
        printf(
            "[INFO]: [Atvoss][Tiling]   R-axis: outer=%lu, inner=%lu, step=%lu, idx=%d\n", SliceDescriptorR_.outer,
            SliceDescriptorR_.inner, SliceDescriptorR_.step, SliceDescriptorR_.idx);
    }


    /*!
     * \brief 计算Reduce后的输出Block大小
     */
    template <class Pattern>
    uint64_t TryGetReduceBlock(
        uint64_t preInputBufferNum, uint64_t preRestBufferNum, uint64_t postBufferNum)
    {
        uint64_t r = CaculateReduceSize<Pattern>();
        // basic * (preMTE * 2 + preVec + 1) + (basic / r) * (postMTE + postVec) = ub - reserved
        uint64_t ubAvilSize = platformInfo_.ubSize - (CACHE_BUF_SIZE + reservedSize_);
        uint64_t ratio = Ratio();

        double basicBlock =
            static_cast<double>(ubAvilSize) / (static_cast<double>(preInputBufferNum + preRestBufferNum * ratio) +
                                            static_cast<double>(postBufferNum * ratio) / static_cast<double>(r));
        return static_cast<uint64_t>(static_cast<double>(basicBlock * ratio) / static_cast<double>(r));
    }

    template <class Pattern>
    uint64_t CaculateReduceSize()
    {
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        int32_t dim = Pattern::TailA ? Pattern::Dim - AXIS_ITERATION_STRIDE : Pattern::Dim - CONST1;
        uint64_t r = 1;
        for (int32_t i = dim; i > -1; i = i - AXIS_ITERATION_STRIDE) {
            if (i == Pattern::Dim - 1) {
                r = r * CeilAlign(shape_[i], ubBlockSize);
            } else {
                r = r * shape_[i];
            }
        }
        return r;
    }

    /*!
     * \brief 计算BasicBlock大小
     *
     * 根据UB可用空间和Reduce/非Reduce比例，计算单次搬运的数据量。
     */
    template <class Pattern>
    void CalcBasicBlock()
    {
        // reduce前的计算图是开启DB的，对于输入输出需要乘以2,
        // 且对于内存策略3而言，reduce算子的输入和计算内存是按照cast前后比例计算
        uint64_t preInputBufferNum = ReducedMTE2Num * CONST2;
        // 用户自定义存活节点在reduce前计算，确保申请足够大的内存
        uint64_t preRestBufferNum = PreReduceTempBufNum + PreReduceMTE3Num * CONST2 + reservedNode_;
        if (PreReduceTempBufNum == 0) {
            // 对于reduce前没有计算的算子，需要额外的inputNum确保搬运流水
            preInputBufferNum = preInputBufferNum * CONST2;
        }
        // reduce后的存活节点加1，用于存放reduce的输出
        uint64_t postBufferNum = ReducedMTE2Num + ReducedTempBufNum + ReducedMTE3Num;

        uint64_t ratio = Ratio();
        uint64_t ubAvilSize = platformInfo_.ubSize - (CACHE_BUF_SIZE + reservedSize_);
        if (Pattern::ID == PATTERN_A) {
            basicBlock_ = ubAvilSize / (preInputBufferNum + (preRestBufferNum + postBufferNum) * ratio);
            basicBlock_ = FloorAlign(basicBlock_, vRegSize);
            resultBlock_ = basicBlock_ * ratio;
        } else {
            uint64_t resBlock = TryGetReduceBlock<Pattern>(preInputBufferNum, preRestBufferNum, postBufferNum);
            resultBlock_ = FloorAlign(resBlock, platformInfo_.cacheLineSize);
            if (resultBlock_ < static_cast<uint64_t>(MAX_INNER_A)) {
                resultBlock_ = MAX_INNER_A;
            } else if (resultBlock_ > CACHE_BUF_SIZE) {
                // 输出大小不能超过cache缓存大小, 否则计算额外A轴搬入时会越界
                resultBlock_ = CACHE_BUF_SIZE;
            }
            if (ubAvilSize <= resultBlock_ * postBufferNum) {
                printf(
                    "[ERROR]: [Atvoss][Tiling] ubSize:%lu is smaller than size:%lu, not support.\n", platformInfo_.ubSize,
                    CACHE_BUF_SIZE + reservedSize_ + resultBlock_ * postBufferNum);
            }
            uint64_t preBufSize = ubAvilSize - resultBlock_ * postBufferNum;
            // 按输入和计算大小的比例折算成输入大小
            basicBlock_ = preBufSize / (preInputBufferNum + preRestBufferNum * ratio);
            basicBlock_ = FloorAlign(basicBlock_, vRegSize);
        }
        if (basicBlock_ < vRegSize) {
            printf("[ERROR]: [Atvoss][Tiling] basic block:%lu is too small, not support.\n", basicBlock_);
        }
        printf(
            "[INFO]: [Atvoss][Tiling] preInputBufferNum:%lu, preRestBufferNum:%lu, postBufferNum:%lu, basicBlock:%ld, resBlock:%ld\n",
            preInputBufferNum, preRestBufferNum, postBufferNum, basicBlock_, resultBlock_);

        return;
    }

    /*!
     * \brief 检查用户Tile配置是否合法（空Tensor场景）
     */
    template <class Pattern>
    bool CheckAndCalEmptyTiling(uint64_t tileA, uint64_t tileR)
    {
        std::vector<uint64_t> AInfo;
        std::vector<uint64_t> RInfo;
        uint64_t baseA = 1;
        uint64_t baseR = 1;
        for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
            if (IsAxisA<Pattern>(dim)) {
                baseA *= shape_[dim];
                AInfo.push_back(baseA);
            } else {
                baseR *= shape_[dim];
                RInfo.push_back(baseR);
            }
        }
        shapeA_ = AInfo[AInfo.size() - 1];
        // A轴为0，直接返回
        if (AInfo[AInfo.size() - 1] == 0) {
            return true;
        }
        // A轴不为0，对A轴进行切分
        if (tileA > AInfo[AInfo.size() - 1]) {
            printf("[ERROR]: [Atvoss][Tiling] tileA exceeds the maximum range of A in the shape. \n");
            return false;
        }
        if (!IsTileMultipleOfLowerDims(AInfo, tileA)) {
            printf("[ERROR]: [Atvoss][Tiling] tileA is invalid; it must be a multiple of the lower dimension. \n");
            return false;
        }
        SetSliceDescriptorWithTileShapeA<Pattern>(tileA);
        return true;
    }

    /*!
     * \brief 检查用户Tile配置是否合法
     *
     * 验证：
     * - Tile大小不超过BasicBlock
     * - Tile不超过对应轴的总大小
     * - Tile是低维轴的整数倍
     */
    template <class Pattern>
    bool CheckTileShape(uint64_t tileA, uint64_t tileR)
    {
        uint64_t tailABlockAlign = 1;
        uint64_t tailRBlockAlign = 1;
        uint64_t tailA = 1;
        uint64_t tailR = 1;
        uint64_t tileABlockAlign = tileA;
        uint64_t tileRBlockAlign = tileR;
        uint64_t maxCacheA = MAX_INNER_A / MaxInputDTypeSize;
        uint64_t maxInnerA = Pattern::ID == PATTERN_A ? basicBlock_ * Ratio() / MaxInputDTypeSize : maxCacheA;
        //计算尾轴block对齐的tileA和tileR
        if (Pattern::TailA) {
            tailA = tileA < shape_[Pattern::Dim - 1] ? tileA : shape_[Pattern::Dim - 1];
            tailABlockAlign = CeilAlign(tailA, platformInfo_.ubBlockSize / inputDtypeSize_);
            tileABlockAlign =
                tileA < shape_[Pattern::Dim - 1] ? tailABlockAlign : tileA / shape_[Pattern::Dim - 1] * tailABlockAlign;
        } else {
            tailR = tileR < shape_[Pattern::Dim - 1] ? tileR : shape_[Pattern::Dim - 1];
            tailRBlockAlign = CeilAlign(tailR, platformInfo_.ubBlockSize / inputDtypeSize_);
            tileRBlockAlign =
                tileR < shape_[Pattern::Dim - 1] ? tailRBlockAlign : tileR / shape_[Pattern::Dim - 1] * tailRBlockAlign;
        }
        // 检查tileshape是否超过basicblock大小
        if (tileABlockAlign * tileRBlockAlign > basicBlock_) {
            printf("[ERROR]: [Atvoss][Tiling] Current tile size exceeds memory limit, please reduce the tile size. \n");
            return false;
        }
        // 检查tileA是否超过A轴缓存最大值
        if (tileABlockAlign > maxInnerA) {
            printf("[ERROR]: [Atvoss][Tiling] tileA exceeds the maximum cache size allowed of axis A. \n");
            return false;
        }
        // 检查tileA和tileR是否满足低维轴的整数倍
        std::vector<uint64_t> AInfo;
        std::vector<uint64_t> RInfo;
        uint64_t baseA = 1;
        uint64_t baseR = 1;
        for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
            if (IsAxisA<Pattern>(dim)) {
                baseA *= shape_[dim];
                AInfo.push_back(baseA);
            } else {
                baseR *= shape_[dim];
                RInfo.push_back(baseR);
            }
        }
        shapeA_ = AInfo[AInfo.size() - 1];
        shapeR_ = RInfo[RInfo.size() - 1];
        if (tileA > AInfo[AInfo.size() - 1]) {
            printf("[ERROR]: [Atvoss][Tiling] tileA exceeds the maximum range of A in the shape. \n");
            return false;
        }
        if (Pattern::ID != PATTERN_A && tileR > RInfo[RInfo.size() - 1]) {
            printf("[ERROR]: [Atvoss][Tiling] tileR exceeds the maximum range of R in the shape. \n");
            return false;
        }
        if (!IsTileMultipleOfLowerDims(AInfo, tileA)) {
            printf("[ERROR]: [Atvoss][Tiling] tileA is invalid; it must be a multiple of the lower dimension. \n");
            return false;
        }
        if (Pattern::ID != PATTERN_A && !IsTileMultipleOfLowerDims(RInfo, tileR)) {
            printf("[ERROR]: [Atvoss][Tiling] tileR is invalid; it must be a multiple of the lower dimension. \n");
            return false;
        }
        return true;
    }

    bool IsTileMultipleOfLowerDims(std::vector<uint64_t> shapeInfo, uint64_t tile)
    {
        uint64_t idx = 0;
        for (uint64_t i = 0; i < shapeInfo.size(); i++) {
            if (tile <= shapeInfo[i]) {
                idx = i;
                break;
            }
        }
        if (idx != 0 && tile % shapeInfo[idx - 1] != 0) {
            return false;
        }
        return true;
    }

    template <class Pattern>
    void SetSliceDescriptorWithTileShapeA(uint64_t tileA)
    {
        int32_t iA;
        uint64_t outerA = shapeA_;
        uint64_t innerA = 1;
        uint64_t step = 1;
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        for (iA = Pattern::TailA ? Pattern::Dim - 1 : Pattern::Dim - 2; iA > -1; iA -= AXIS_ITERATION_STRIDE) {
            uint64_t axisLen = shape_[iA];
            outerA /= axisLen;
            if (tileA <= axisLen) {
                innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(tileA, ubBlockSize) : tileA);
                outerA *= CeilDiv(axisLen, tileA);
                step = tileA;
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            tileA /= axisLen;
        }
        SliceDescriptorA_.AssembleUnit(iA, innerA, outerA, step);
    }

    template <class Pattern>
    void SetSliceDescriptorWithTileShapeR(uint64_t tileR)
    {
        int32_t iR;
        uint64_t outerR = shapeR_;
        uint64_t innerR = 1;
        uint64_t step = 1;
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        for (iR = Pattern::TailA ? Pattern::Dim - 2 : Pattern::Dim - 1; iR > -1; iR -= AXIS_ITERATION_STRIDE) {
            uint64_t axisLen = shape_[iR];
            outerR /= axisLen;
            if (tileR <= axisLen) {
                innerR *= (iR == Pattern::Dim - 1 ? CeilAlign(tileR, ubBlockSize) : tileR);
                outerR *= CeilDiv(axisLen, tileR);
                step = tileR;
                break;
            }
            innerR *= (iR == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            tileR = tileR / axisLen;
        }
        SliceDescriptorR_.AssembleUnit(iR, innerR, outerR, step);
    }

    template <class Pattern>
    bool IsEmtpyTensor()
    {
        for (int32_t i = 0; i < Pattern::Dim; i++) {
            if (shape_[i] == 0) {
                return true;
            }
        }
        return false;
    }

    /*!
     * \brief 计算CacheLine切分
     *
     * 根据CacheLine大小确定搬运切分位置，优化GM到UB的搬运效率。
     */
    template <class Pattern>
    void ComputeCacheLineBlock()
    {
        uint64_t cacheSize = platformInfo_.cacheLineSize / inputDtypeSize_;
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        uint64_t cacheLineShape = 1UL;
        uint64_t cacheLineStep = 1UL;
        uint64_t cacheLineOuter = 1UL;
        uint64_t aInCacheLine = 1UL;
        uint64_t rInCacheLine = 1UL;

        for (int32_t i = Pattern::Dim - 1; i > -1; --i) {
            cacheLineShape *= shape_[i];
            if (cacheLineShape > cacheSize) {
                cacheLineShape /= shape_[i];
                cacheLineStep = CeilDiv(cacheSize, cacheLineShape);
                cacheLineOuter = CeilDiv(shape_[i], cacheLineStep);
                cBlock_.axis = i;
                break;
            } else {
                cacheLineStep = shape_[i];
                cBlock_.axis = i;
            }
        }

        for (int32_t i = Pattern::Dim - 1; i > cBlock_.axis; --i) {
            if (i == Pattern::Dim - 1) {
                if (IsAxisA<Pattern>(i)) {
                    aInCacheLine = aInCacheLine * CeilAlign(shape_[i], ubBlockSize);
                } else {
                    rInCacheLine = rInCacheLine * CeilAlign(shape_[i], ubBlockSize);
                }
            } else {
                if (IsAxisA<Pattern>(i)) {
                    aInCacheLine = aInCacheLine * shape_[i];
                } else {
                    rInCacheLine = rInCacheLine * shape_[i];
                }
            }
        }

        cBlock_.step = cacheLineStep;
        cBlock_.stepCount = cacheLineOuter;
        cBlock_.aSize = aInCacheLine;
        cBlock_.rSize = rInCacheLine;
        printf(
            "[INFO]: [Atvoss][Tiling] cacheLine Block axis:%d, cacheLineStep:%lu, cacheLineOuter:%lu, aSize:%lu, rSize:%lu\n", cBlock_.axis,
            cBlock_.step, cBlock_.stepCount, cBlock_.aSize, cBlock_.rSize);
    }

    template <class Pattern>
    void InitUnit()
    {
        int32_t axisInCacheLine = cBlock_.axis;
        for (int32_t i = Pattern::FirstA ? 0 : 1; i < axisInCacheLine; i += AXIS_ITERATION_STRIDE) {
            SliceDescriptorA_.outer *= shape_[i];
        }
        for (int32_t i = Pattern::FirstA ? 1 : 0; i < axisInCacheLine; i += AXIS_ITERATION_STRIDE) {
            SliceDescriptorR_.outer *= shape_[i];
        }
        bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
        if (basicSplitA) {
            SliceDescriptorA_.outer *= cBlock_.stepCount;
        } else {
            SliceDescriptorR_.outer *= cBlock_.stepCount;
        }
    }

    template <class Pattern>
    void ComputeCacheLineBlockAndUnit()
    {
        ComputeCacheLineBlock<Pattern>();
        InitUnit<Pattern>();
    }

    /*
     * 计算UB内A轴的切分大小，最大512B或者按A轴分核低于85%
     * 例如: float32, shape:(12800, 8), pattern:AR, cacheLine:256B, coreNum:64
     * cacheLine切分后(1600, cacheLine(8, 8))
     * 对1600做A轴切分，找到uintA inner:16(cacheline中A轴为8, 与16相乘后达到512B上限), outer:100
     */
    template <class Pattern>
    void ComputeUnitA()
    {
        //初始化相关参数
        int32_t axisInCacheLine = cBlock_.axis;
        bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
        uint64_t outerA = (basicSplitA ? SliceDescriptorA_.outer / cBlock_.stepCount * shape_[axisInCacheLine] : SliceDescriptorA_.outer);
        uint64_t innerA = SliceDescriptorA_.inner;
        uint64_t bBlockNum = basicBlock_ * Ratio() / MaxInputDTypeSize;
        uint64_t totalRSize = CaculateReduceSize<Pattern>();
        uint64_t aInCacheLineSize = basicSplitA ? cBlock_.aSize * cBlock_.stepCount : cBlock_.aSize;
        //如果除去cacheline内的元素数量，R轴全载仍能剩余2个及以上的A，则R轴全载，A轴切分停留在cacheline切分
        bool rFullLoad = (bBlockNum / aInCacheLineSize / totalRSize >= 2);
        int32_t iAStart = basicSplitA ? axisInCacheLine : axisInCacheLine - 1;
        if (rFullLoad) {
            int32_t iA = iAStart;
            uint64_t step = (iA == axisInCacheLine ? cBlock_.stepCount : 1UL);
            innerA = innerA * step;
            outerA = outerA / shape_[iA] * CeilDiv(shape_[iA], step);
            SliceDescriptorA_.AssembleUnit(iA, innerA, outerA, step);
            return;
        } else {
            FindUnitASplitAxis<Pattern>(basicSplitA, iAStart, innerA, outerA, bBlockNum);
        }
    }

    template <class Pattern>
    void FindUnitASplitAxis(bool basicSplitA, int32_t iAStart, uint64_t innerA, uint64_t outerA, uint64_t bBlockNum)
    {
        uint64_t maxCacheA = MAX_INNER_A / MaxInputDTypeSize;
        uint64_t maxInnerA = Pattern::ID == PATTERN_A ? basicBlock_ * Ratio() / MaxInputDTypeSize : maxCacheA;
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        uint64_t step = 1;
        int32_t iA = iAStart;
        int32_t lastAxisA = Pattern::TailA ? (Pattern::Dim - 1) : (Pattern::Dim - 2);
        for (; iA > -1; iA -= AXIS_ITERATION_STRIDE) {
            uint64_t axisLen = shape_[iA];
            uint64_t stepLen = 1;
            uint64_t maxStep = 1;
            uint64_t innerAOrigin = calOriginInnerA<Pattern>(iA);
            // 根据不同场景，初始化step和步长
            InitStepAndSteplen(&step, &stepLen, &maxStep, iA, lastAxisA, cBlock_.axis, innerAOrigin);
            bool splitHere = false;
            if (step > axisLen) {
                maxStep = 1;
            }
            //该根轴上寻找合适的ubfactorA
            //RA场景，满足基本分核数的情况下多进A；AR场景，满足基本分核数就结束
            for (; step <= axisLen; step = step + stepLen) {
                uint64_t s = step;
                uint64_t tmpInnerA = innerA * s;
                uint64_t tmpOuterA = outerA / axisLen * CeilDiv(shape_[iA], s);
                uint64_t aSize = tmpInnerA * cBlock_.aSize;
                if (aSize <= maxInnerA && aSize * cBlock_.rSize <= bBlockNum) {
                    uint64_t tempCoreNum = tmpOuterA / CeilDiv(tmpOuterA, platformInfo_.vectorCoreNum);
                    tempCoreNum = tempCoreNum > tmpOuterA ? FloorAlign(tempCoreNum, tmpOuterA) : tempCoreNum;
                    if (Pattern::TailA) {
                        maxStep = (tempCoreNum > THRES_CORENUM) ? s : maxStep;
                    } else {
                        if (tempCoreNum > THRES_CORENUM) {
                            maxStep = s;
                            splitHere = true;
                            break;
                        }
                    }
                } else {
                    splitHere = true;
                    break;
                }
            }
            if (splitHere || maxStep != axisLen || iA - AXIS_ITERATION_STRIDE < 0) {
                // A轴最大、分核最大或者无法继续迭代
                step = maxStep;
                innerA = innerA * step;
                outerA = outerA / axisLen * CeilDiv(shape_[iA], step);
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            outerA /= axisLen;
        }
        SliceDescriptorA_.AssembleUnit(iA, innerA, outerA, step);
    }
    /*
     * 如果当前处理是cacheline切分轴：
     * 如果当前处理最后一根A轴，步长为cacheSize，初始step为cachelinestep和cachesize的最大值；
     * 如果当前处理不是最后一根A轴，根据cacheline切分内A轴乘积是奇数还是偶数分开step和步长为奇数还是偶数。
     * 如果当前处理不是cacheline切分轴：
     * 如果当前处理最后一根A轴，初始step和步长为cacheSize；
     * 如果当前处理不是最后一根A轴，根据cacheline切分内A轴乘积是奇数还是偶数分开step和步长为1还是2。
     */
    void InitStepAndSteplen(uint64_t* Astep, uint64_t* AstepLen, uint64_t* AmaxStep, int32_t iA, int32_t lastAxisA, int32_t axisInCacheLine, uint64_t innerAOrigin)
    {
        uint64_t cacheSize = platformInfo_.cacheLineSize / inputDtypeSize_ / 2;
        bool isLastAxisA = (iA == lastAxisA);
        //判断cacheline切分内A轴乘积是奇数还是偶数
        bool parityFactor = (cBlock_.aSize != 1 && innerAOrigin % 2 == 1);
        //判断是否是cacheline切分轴
        if (iA == axisInCacheLine) {
            if (isLastAxisA) {
                *Astep = cBlock_.step < cacheSize ? cacheSize : cBlock_.step;
                *AstepLen = cacheSize;
            } else {
                *Astep = parityFactor ? CeilAlign(cBlock_.step, static_cast<uint64_t>(2)) : cBlock_.step;
                *AstepLen = parityFactor ? 2UL : 1UL;
            }
        } else {
            if (isLastAxisA) {
                *Astep = cacheSize;
                *AstepLen = cacheSize;
            } else {
                *Astep = parityFactor ? 2UL : 1UL;
                *AstepLen = parityFactor ? 2UL : 1UL;
            }
        }
        *AmaxStep = *Astep;
    }
    /*
     * 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
     * 用BasicBlock / UB内A的切分大小，找到满BasicBlock的R轴切分位置
     */
    template <class Pattern>
    void ComputeUnitR()
    {
        //初始化
        int32_t axisInCacheLine = cBlock_.axis;
        bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
        uint64_t outerR = (basicSplitA ? SliceDescriptorR_.outer : SliceDescriptorR_.outer / cBlock_.stepCount * shape_[axisInCacheLine]);
        uint64_t innerR = SliceDescriptorR_.inner;
        uint64_t outerA = SliceDescriptorA_.outer;
        uint64_t innerA = SliceDescriptorA_.inner;
        uint64_t step = 1UL;
        uint64_t ubFactorR = 1UL;
        uint64_t bBlockNum = basicBlock_ * Ratio() / MaxInputDTypeSize;
        uint64_t cacheSize = platformInfo_.cacheLineSize / inputDtypeSize_ / 2;
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        int32_t iR;
        //找切分轴，原则：找到ub用满的轴
        for (iR = basicSplitA ? axisInCacheLine - 1 : axisInCacheLine; iR > -1; iR -= AXIS_ITERATION_STRIDE) {
            uint64_t axisLen = shape_[iR];
            if (innerA * innerR * cBlock_.aSize * cBlock_.rSize * axisLen <= bBlockNum) {
                outerR = outerR / axisLen;
                innerR *= (iR == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
                continue;
            }

            // maybe bBlockNum not full
            step = std::min(bBlockNum / (innerA * innerR * cBlock_.aSize * cBlock_.rSize), axisLen);
            if (iR == Pattern::Dim - 1) {
                step = FloorAlign(step, ubBlockSize);
            }
            uint64_t minStep = (iR == axisInCacheLine ? cBlock_.step : 1UL);
            uint64_t stepLen = (iR == Pattern::Dim - 1) ? cacheSize : 1;
            bool is_prime = isPrime(axisLen);
            //如果切分轴是尾轴或者切分轴是质数不能被均分，正常切分；否则按照均分切分轴切分
            if (iR != Pattern::Dim - 1 && !is_prime) {
                ubFactorR = findDivisibleUbFactorR<Pattern>(step, minStep, stepLen, axisLen, outerA, outerR, ubBlockSize, iR);
            } else {
                ubFactorR = findUbFactorR<Pattern>(step, minStep, stepLen, axisLen, outerA, outerR, ubBlockSize, iR);
            }
            //在当前轴找满足分核数的ubfactorR
            // R轴非尾轴时，继续遍历寻找能够均分切分轴的R
            innerR = innerR * ubFactorR;
            outerR = outerR / axisLen * CeilDiv(shape_[iR], ubFactorR);
            break;
        }
        SliceDescriptorR_.AssembleUnit(iR, innerR, outerR, step);;
    }

    template <class Pattern>
    uint64_t findUbFactorR(uint64_t step, uint64_t minStep, uint64_t stepLen, uint64_t axisLen, uint64_t outerA, uint64_t outerR, uint64_t ubBlockSize, int32_t iR)
    {
        uint64_t s = step;
        for (; s > minStep; s = s - stepLen) {
            uint64_t tmpS = s;
            if (iR == Pattern::Dim - 1) {
                // 尾轴R场景，尽可能保证尾轴切分block对齐，减少kernel侧补pad
                tmpS = FloorAlign(tmpS, ubBlockSize);
            }
            auto tmpOuterR = outerR / axisLen * CeilDiv(shape_[iR], tmpS);

            uint64_t tempCoreNum = (outerA * tmpOuterR) / CeilDiv(outerA * tmpOuterR, platformInfo_.vectorCoreNum);
            tempCoreNum = tempCoreNum > outerA ? FloorAlign(tempCoreNum, outerA) : tempCoreNum;
            if (tempCoreNum > THRES_CORENUM) {
                // 找不到合适的rate，就不刷新step
                return tmpS;
            }
        }
        return step;
    }

    template <class Pattern>
    uint64_t findDivisibleUbFactorR(uint64_t step, uint64_t minStep, uint64_t stepLen, uint64_t axisLen, uint64_t outerA, uint64_t outerR, uint64_t ubBlockSize, int32_t iR)
    {
        uint64_t s = step;
        for (; s > minStep; s = s - stepLen) {
            if (axisLen % s == 0){
                uint64_t tmpS = s;
                if (iR == Pattern::Dim - 1) {
                    // 尾轴R场景，尽可能保证尾轴切分block对齐，减少kernel侧补pad
                    tmpS = FloorAlign(tmpS, ubBlockSize);
                }
                auto tmpOuterR = outerR / axisLen * CeilDiv(shape_[iR], tmpS);

                uint64_t tempCoreNum = (outerA * tmpOuterR) / CeilDiv(outerA * tmpOuterR, platformInfo_.vectorCoreNum);
                tempCoreNum = tempCoreNum > outerA ? FloorAlign(tempCoreNum, outerA) : tempCoreNum;
                if (tempCoreNum > THRES_CORENUM) {
                    // 找不到合适的rate，就不刷新step
                    return tmpS;
                }
            }
        }
        return step;
    }
    /*
     * 针对R轴过小，UB内R轴全载，BasicBlock不能满载时，调整UB内A轴切分，上限Reduce计算后输出buffer大小
     */
    template <class Pattern>
    void ComputeProgressUnitA()
    {
        // if RAxis is fully loaded in UB, and basicBlock is not enough, we need to calculate extra unitA
        if (SliceDescriptorR_.idx != -1) {
            return;
        }
        uint64_t innerA = SliceDescriptorA_.inner / SliceDescriptorA_.step; // restore original innerA
        uint64_t outerA = SliceDescriptorA_.outer / CeilDiv(shape_[SliceDescriptorA_.idx], SliceDescriptorA_.step) * shape_[SliceDescriptorA_.idx];
        uint64_t bBlockNum = basicBlock_ * Ratio() / MaxInputDTypeSize;
        uint64_t maxInnerA = resultBlock_ / MaxInputDTypeSize;
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        uint64_t cacheSize = platformInfo_.cacheLineSize / inputDtypeSize_;
        uint64_t innerR = SliceDescriptorR_.inner;
        uint64_t step = 1;
        int32_t iA;
        for (iA = SliceDescriptorA_.idx; iA > -1; iA -= AXIS_ITERATION_STRIDE) {
            uint64_t axisLen = shape_[iA];
            uint64_t stepLen = 1;
            uint64_t maxStep = 1;
            InitStepAndSteplenFullLoadR(&step, &stepLen, &maxStep, iA);
            bool splitHere = false;
            uint64_t maxCoreNum = 1;
            for (; step <= axisLen; step = step + stepLen) {
                uint64_t tmpS = step;
                if (iA == Pattern::Dim - 1) {
                    tmpS = CeilAlign(tmpS, ubBlockSize);
                }
                uint64_t tmpInnerA = innerA * tmpS;
                uint64_t tmpOuterA = outerA / axisLen * CeilDiv(shape_[iA], tmpS);
                uint64_t tempCoreNum = tmpOuterA / CeilDiv(tmpOuterA, platformInfo_.vectorCoreNum);
                tempCoreNum = tempCoreNum > tmpOuterA ? FloorAlign(tempCoreNum, tmpOuterA) : tempCoreNum;
                bool isContinue =
                    (tmpInnerA * innerR * cBlock_.aSize * cBlock_.rSize <= bBlockNum &&
                     tmpInnerA * cBlock_.aSize <= maxInnerA);
                if (isContinue) {
                    if (tmpInnerA * cBlock_.aSize <= cacheSize) {
                        maxStep = tmpS;
                        maxCoreNum = tempCoreNum > maxCoreNum ? tempCoreNum : maxCoreNum;
                    } else if (tempCoreNum >= THRES_CORENUM && tempCoreNum >= maxCoreRatio * maxCoreNum) {
                        maxStep = tmpS;
                        maxCoreNum = tempCoreNum > maxCoreNum ? tempCoreNum : maxCoreNum;
                    }
                    continue;
                } else {
                    splitHere = true;
                    break;
                }
            }
            if (splitHere || iA - AXIS_ITERATION_STRIDE < 0) {
                step = maxStep;
                innerA *= step;
                outerA = outerA / axisLen * CeilDiv(shape_[iA], step);
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            outerA /= axisLen;
        }
        SliceDescriptorA_.AssembleUnit(iA, innerA, outerA, step);
    }

    void InitStepAndSteplenFullLoadR(uint64_t* Astep, uint64_t* AstepLen, uint64_t* maxStepA, int32_t iA)
    {
        if (iA == SliceDescriptorA_.idx) {
            *Astep = SliceDescriptorA_.step + 1UL;
            *AstepLen = 1UL;
            *maxStepA = SliceDescriptorA_.step;
        } else {
            *Astep = 1UL;
            *AstepLen = 1UL;
            *maxStepA = 1;
        }
    }


    /*!
     * \brief Shape变换：消除1维和合并连续轴
     *
     * Reduce算子需要对Shape进行预处理：
     * 1. EliminateOne - 消除大小为1且连续的维度
     * 2. MergeAxis - 合并连续的同类型轴（连续A轴或连续R轴）
     *
     * 变换后形成A/R交替的Pattern，便于Tiling计算。
     */
    void TransformShape(const std::vector<uint64_t>& shapeInfo, std::vector<int>& axes)
    {
        EliminateOne(shapeInfo, axes);
        MergeAxis(axes);
    }

    /*!
     * \brief 消除大小为1且内存连续的维度
     */
    void EliminateOne(const std::vector<uint64_t>& shapeInfo, std::vector<int>& axes)
    {
        shape_[0] = 1UL;
        int32_t dstIdx = 1;
        for (size_t i = 0; i < axes.size(); i++) {
            axes[i] = axes[i] + 1;
        }
        int32_t eraseNum = 0;
        for (size_t i = 0; i < shapeInfo.size(); i++) {
            auto iter = std::find(axes.begin(), axes.end(), static_cast<int>(i + 1));
            if (shapeInfo[i] != 1L) {
                shape_[dstIdx] = shapeInfo[i];
                if (iter != axes.end()) {
                    *iter = *iter - eraseNum;
                }
                dstIdx++;
            } else {
                eraseNum++;
                if (iter != axes.end()) {
                    axes.erase(iter);
                }
            }
        }
        newShapeSize_ = dstIdx;
        printf(
            "[INFO]: [Atvoss][Tiling] after EliminateOne, Shape is:%s, axes:%s\n", ToString(shape_, newShapeSize_).c_str(),
            ToString(axes).c_str());
    }

    /*!
     * \brief 合并连续的同类型轴（A轴或R轴）
     *
     * 将连续的多个A轴合并为一个A轴，连续的多个R轴合并为一个R轴，
     * 形成A/R交替的Pattern结构。
     */
    void MergeAxis(std::vector<int>& axes)
    {
        int32_t tmpSize = 0;
        for (int32_t i = 0; i < newShapeSize_;) {
            auto iter0 = std::find(axes.begin(), axes.end(), i);
            bool isRAxis0 = iter0 != axes.end();
            uint64_t s = shape_[i];
            int32_t j = i + 1;
            for (; j < newShapeSize_; j++) {
                auto iter1 = std::find(axes.begin(), axes.end(), j);
                bool isRAxis1 = iter1 != axes.end();
                if (isRAxis0 != isRAxis1) {
                    break;
                }
                s *= shape_[j];
                if (isRAxis1) {
                    axes.erase(iter1);
                }
            }
            i = j;
            shape_[tmpSize] = s;
            tmpSize++;
            if (isRAxis0) {
                *iter0 = tmpSize - 1;
            }
        }
        for (int32_t i = tmpSize; i < newShapeSize_; i++) {
            shape_[i] = 0;
        }
        newShapeSize_ = tmpSize;
        printf(
            "[INFO]: [Atvoss][Tiling] after merge axis, Shape:%s, axes:%s\n", ToString(shape_, newShapeSize_).c_str(),
            ToString(axes).c_str());
    }

    /*!
     * \brief 计算Shape元素总数
     */
    uint64_t CalcShapeSize(const std::vector<uint64_t>& shape)
    {
        uint64_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        return size;
    }

    /*!
     * \brief 遍历tuple中的单个Tensor，获取其shape并计算size
     */
    template <typename TensorType, size_t Index, typename TupleType>
    void GetTensorShapeInfo(const TupleType& tuple, std::vector<std::vector<uint64_t>>& allShapes)
    {
        const auto& tensor = std::get<Index>(tuple);
        allShapes.push_back(tensor.shape_vector());
    }

    /*!
     * \brief 遍历整个tuple，收集所有Tensor的shape信息
     */
    template <typename TupleType, size_t... Indices>
    void ExtractInputOutputShapes(
        const TupleType& tuple, std::vector<std::vector<uint64_t>>& allShapes, std::index_sequence<Indices...>)
    {
        (GetTensorShapeInfo<typename std::tuple_element_t<Indices, TupleType>, Indices, TupleType>(tuple, allShapes),
         ...);
    }

    /*!
     * \brief 校验输入输出shape的正确性
     *
     * 1. 遍历tuple，找shape size最大的作为输入shape
     * 2. 如果出现三个及以上不同的shape size，报错
     * 3. 将shape size较大的作为reduce前shape，较小的作为reduce后shape
     */
    template <typename TupleType>
    bool ValidateInputOutputShapes(
        const TupleType& tensorTuple, std::vector<uint64_t>& inputShape, std::vector<uint64_t>& outputShape)
    {
        constexpr size_t tupleSize = std::tuple_size_v<TupleType>;
        std::vector<std::vector<uint64_t>> allShapes;
        ExtractInputOutputShapes(tensorTuple, allShapes, std::make_index_sequence<tupleSize>{});

        if (allShapes.empty()) {
            printf("[ERROR]: [Atvoss][Tiling] No tensor found in arguments\n");
            return false;
        }

        std::vector<uint64_t> shapeSizes;
        for (const auto& shape : allShapes) {
            shapeSizes.push_back(CalcShapeSize(shape));
        }

        std::set<uint64_t> uniqueSizes(shapeSizes.begin(), shapeSizes.end());
        if (uniqueSizes.size() > 2) {
            printf("[ERROR]: [Atvoss][Tiling] 输入输出shape设置有误，请检查。检测到超过2种不同的shape size\n");
            return false;
        }

        size_t maxIndex = 0;
        size_t minIndex = 0;
        uint64_t maxSize = shapeSizes[0];
        uint64_t minSize = shapeSizes[0];

        for (size_t i = 1; i < shapeSizes.size(); i++) {
            if (shapeSizes[i] > maxSize) {
                maxSize = shapeSizes[i];
                maxIndex = i;
            }
            if (shapeSizes[i] < minSize) {
                minSize = shapeSizes[i];
                minIndex = i;
            }
        }

        inputShape = allShapes[maxIndex];
        outputShape = allShapes[minIndex];

        for (size_t i = 0; i < allShapes.size(); i++) {
            uint64_t currentSize = CalcShapeSize(allShapes[i]);
            if (currentSize == maxSize && allShapes[i] != inputShape) {
                printf("[ERROR]: [Atvoss][Tiling] Tensor %zu has same size but different shape than input tensor\n", i);
                return false;
            }
            if (currentSize == minSize && allShapes[i] != outputShape) {
                printf(
                    "[ERROR]: [Atvoss][Tiling] Tensor %zu has same size but different shape than output tensor\n", i);
                return false;
            }
        }

        printf("[INFO]: [Atvoss][Tiling] Input shape size: %lu, Output shape size: %lu\n", maxSize, minSize);
        return true;
    }

    /*!
     * \brief 校验reduce后的output shape是否正确
     *
     * 根据reduce前的input shape和reduce轴axes，以及keep_dim参数，
     * 计算预期的output shape，并与实际的output shape进行比对。
     *
     * \param[in] inputShape reduce前的输入shape
     * \param[in] axes reduce轴列表
     * \param[in] outputShape 实际的输出shape
     * \param[in] keepDim 是否保留reduce轴维度（true: 保留为1，false: 移除该轴）
     * \return true 校验通过，false 校验失败
     */
    bool ValidateReduceOutputShape(
        const std::vector<uint64_t>& inputShape, const std::vector<int>& axes,
        const std::vector<uint64_t>& outputShape)
    {
        bool keepDim = true;
        size_t inputDim = inputShape.size();
        size_t outputDim = outputShape.size();
        size_t axesNum = axes.size();

        if (outputDim == inputDim) {
            keepDim = true;
        } else if (outputDim == inputDim - axesNum) {
            keepDim = false;
        } else {
            printf(
                "[ERROR]: [Atvoss][Tiling] Output shape dimension invalid: input %zu, axes %zu, output %zu\n", inputDim,
                axesNum, outputDim);
            return false;
        }

        std::set<int> reduceAxes(axes.begin(), axes.end());

        std::vector<uint64_t> expectedOutputShape;
        for (size_t i = 0; i < inputShape.size(); i++) {
            if (reduceAxes.count(static_cast<int>(i)) > 0) {
                if (keepDim) {
                    expectedOutputShape.push_back(1);
                }
            } else {
                expectedOutputShape.push_back(inputShape[i]);
            }
        }

        if (expectedOutputShape.size() != outputShape.size()) {
            printf(
                "[ERROR]: [Atvoss][Tiling] Output shape dimension mismatch: expected %zu, got %zu (keep_dim=%s)\n",
                expectedOutputShape.size(), outputShape.size(), keepDim ? "true" : "false");
            return false;
        }

        for (size_t i = 0; i < outputShape.size(); i++) {
            if (expectedOutputShape[i] != outputShape[i]) {
                printf(
                    "[ERROR]: [Atvoss][Tiling] Output shape mismatch at axis %zu: expected %lu, got %lu\n", i,
                    expectedOutputShape[i], outputShape[i]);
                return false;
            }
        }
        return true;
    }


    template <class Pattern>
    int32_t IsUseNddma()
    {
        int32_t axis = cBlock_.axis;
        uint64_t ubBlockSize = platformInfo_.ubBlockSize / inputDtypeSize_;
        if (shape_[Pattern::Dim - 1] >= ubBlockSize) {
            // last dim 大于ubblock, 不做NDDMA
            return 0;
        }
        if ((Pattern::Dim - 1 - axis > CONST2) || (Pattern::Dim - 1 - axis == CONST2 && cBlock_.step != 1UL)) {
            // cacheline切分超过3维，或者等于三维时，最高维不为1。即cacheline有效切分维度超过2维
            return 1;
        }
        if (Pattern::TailA) {
            uint64_t factorA = SliceDescriptorA_.step;
            for (auto iA = SliceDescriptorA_.idx + AXIS_ITERATION_STRIDE; iA < Pattern::Dim; iA += AXIS_ITERATION_STRIDE) {
                factorA = factorA * shape_[iA];
            }
            if (factorA > ubBlockSize) {
                // 转置后A轴乘积大于ubblock, 不做NDDMA
                return 0;
            }
        } else {
            uint64_t factorR = SliceDescriptorR_.step;
            for (auto iR = SliceDescriptorR_.idx + AXIS_ITERATION_STRIDE; iR < Pattern::Dim; iR += AXIS_ITERATION_STRIDE) {
                factorR = factorR * shape_[iR];
            }
            if (factorR > ubBlockSize) {
                // 转置后R轴乘积大于ubblock, 不做NDDMA
                return 0;
            }
        }
        return 1;
    }

    template <class Pattern>
    void ComputeStride()
    {
        uint64_t s = 1UL;
        uint64_t ds = 1UL;
        for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
            srcStride_[dim] = s;
            outStride_[dim] = ds;
            s *= shape_[dim];
            if (IsAxisA<Pattern>(dim)) {
                ds *= shape_[dim];
            }
        }
    }

    uint64_t Ratio()
    {
        // if (MemLevel::LEVEL_0) {
        //     return CeilDiv(MaxInputDTypeSize, static_cast<uint64_t>(inputDtypeSize_));
        // }
        return 1UL;
    }

    template <class Pattern>
    bool IsAxisA(int32_t idx)
    {
        if (Pattern::FirstA) {
            return idx % CONST2 == 0;
        } else {
            return idx % CONST2 == 1;
        }
    }

    template <class Pattern>
    uint64_t calOriginInnerA(int32_t axis)
    {
        uint64_t innerAOrigin = 1;
        for (int32_t i = Pattern::TailA ? Pattern::Dim - 1 : Pattern::Dim - 2; i > axis; i -= AXIS_ITERATION_STRIDE) {
            innerAOrigin *= shape_[i];
        }
        return innerAOrigin;
    }

    bool isPrime(uint64_t num) {
        if (num <= CONST1) return false;
        if (num == CONST2 || num == CONST3) return true;
        if (num % CONST2 == 0 || num % CONST3 == 0) return false;

        for (uint64_t i = CONST5; i * i <= num; i += CONST6) {
            if (num % i == 0 || num % (i + CONST2) == 0) {
                return false;
            }
        }
        return true;
    }
#if !defined(__ATVOSS_HOST_ONLY__)
public:
    /*!
     * \brief Kernel层构造函数
     */
    __host_aicore__ inline BaseKernelSchedule() = default;

    /*!
     * \brief Kernel层执行入口
     *
     * 根据BlockIdx计算当前Core需要处理的数据范围，
     * 构造Block参数，调用Block执行。
     *
     * \param[in] cfg Kernel配置参数（包含tilingData和blockParam）
     * \param[in] args 输入输出GM地址
     */
    template <auto SchLoopInfo, typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        typename BlockOp::ScheduleClz::ParamStruct blockConfig = cfg.tilingData;
        // in and out gm address
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);

        // 根据当前coreNum计算gm起始地址, 按 DAG 阶段自动区分 PreReduce / PostReduce
        auto params = KernelParamHandler::PrepareParamsByPhase<
            PreReduceParams, PostReduceParams>(cfg.tilingData, argTuple);
        // args和params参数合并
        auto convertArgs = KernelParamHandler::ConvertArgs<Params>(params, argTuple);

        if constexpr (SchLoopInfo->loopRCount == 0) {
            BlockOp blockOp{};
            blockOp.template Run<SchLoopInfo>(blockConfig, convertArgs);
        } else {
            BlockOp blockOpStepOne{};
            blockOpStepOne.template Run<SchLoopInfo>(blockConfig, convertArgs);
            AscendC::SyncAll();
            TilingForGroup(cfg);
            BlockOp blockOpStepTwo{};
            blockOpStepTwo.template Run<SchLoopInfo>(blockConfig, convertArgs);
        }
    }

    /**
     * @brief Group Reduce 的 Tiling 处理
     * @tparam OpParam 操作参数类型
     * @param cfg Tiling 配置参数
     */
    template <typename OpParam>
    __aicore__ inline auto TilingForGroup(OpParam& cfg)
    {
        return;
    }
#endif
};

/*!
 * \brief 默认Kernel调度类，继承BaseKernelSchedule
 */
template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class DefaultKernelSchedule : public BaseKernelSchedule<BlockOp, Policy, ScheduleCfg> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_KERNEL_SCHEDULE_H_
