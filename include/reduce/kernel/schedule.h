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
 * 3. 计算A轴和R轴的切分参数（unitA_, unitR_）
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
#include "common/arch.h"
#include "common/compile_info.h"
#include "reduce/reduce_util.h"
#include "device/tilingdata.h"
#include "graph/dag.h"
#include "expression/expr_template.h"

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
};

/*!
 * \brief CacheLine切分块，记录CacheLine级别的切分参数
 *
 * 为了保证GM到UB的数据搬运效率，需要按CacheLine大小切分，
 * 减少搬运次数，提升带宽利用率。
 */
struct CacheLineSliceInfo {
    int32_t axis = -1;           // CacheLine切分轴
    uint64_t cacheLineStep = 1;  // CacheLine步长
    uint64_t cacheLineOuter = 1; // CacheLine外层迭代次数
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
    using Params = Atvoss::Params_t<Expr>;                    // 参数列表
    using TileShape = typename BlockTemplate::BlockTileShape; // Tile形状
    using ArchTag = typename BlockOp::ScheduleClz::ArchTag;   // 架构标签
    using BlockPolicy = typename BlockOp::BlockPolicy;
    static Atvoss::OpPlatformInfo compileInfo_;                // 编译信息

    // ========== 从表达式推导 DAG 信息 ==========
    // 将 OpAndThen 展开为 TypeList，供 ReduceAutoDag 使用
    using ExprList = typename FlattenAtOpAndThen<Expr>::Type;
    // 是否启用二分累加（从 BlockPolicy 获取）
    static constexpr bool EnableBinaryAcc = BlockOp::policy.EnableBinaryAcc;
    // 定义 DAG 类型
    using DagType = Graph::ReduceAutoDag<ExprList, EnableBinaryAcc>;

    // ========== DAG 静态信息（编译期常量） ==========
    // PreReduce 阶段 MTE2 数量（GM → UB 拷入次数）
    static constexpr size_t PreReduceMTE2Num = DagType::GetMTE2Num<true>();
    // Reduced 阶段 MTE2 数量
    static constexpr size_t ReducedMTE2Num = DagType::GetMTE2Num<false>();
    // PreReduce 阶段 MTE3 数量（UB → GM 拷出次数）
    static constexpr size_t PreReduceMTE3Num = DagType::GetMTE3Num<true>();
    // Reduced 阶段 MTE3 数量
    static constexpr size_t ReducedMTE3Num = DagType::GetMTE3Num<false>();
    // PreReduce 临时 buffer 数量
    static constexpr int32_t PreReduceTempBufNum = DagType::GetTempBufNum<true>();
    // Reduced 临时 buffer 数量
    static constexpr int32_t ReducedTempBufNum = DagType::GetTempBufNum<false>();
    // 输入最大 dtype 字节数
    static constexpr int32_t MaxInputDTypeSize = DagType::GetInputMaxDTypeSize<true>();
    // 输入最小 dtype 字节数
    static constexpr int32_t MinInputDTypeSize = DagType::GetInputMinDTypeSize<true>();
    // 最大 dtype 字节数（输入、输出、临时变量的最大值）
    static constexpr int32_t MaxDTypeSize = DagType::GetMaxDTypeSize<true>();
    // 最小 dtype 字节数
    static constexpr int32_t MinDTypeSize = DagType::GetMinDTypeSize<true>();


    // ========== 常量配置 ==========
    static constexpr uint64_t TILEA = TileShape::template get_type<0>::value; // 用户配置的A轴Tile大小
    static constexpr uint64_t TILER = TileShape::template get_type<1>::value; // 用户配置的R轴Tile大小
    static constexpr int32_t MAX_INNER_A = 512;                               // A轴UB内最大512字节
    static constexpr int32_t MEMBOUND_CORENUM = 40;                           // MemoryBound场景核心数阈值
    static constexpr int32_t VECBOUND_CORENUM = 48;                           // VectorBound场景核心数阈值
    static constexpr int32_t THRES_CORENUM = MEMBOUND_CORENUM;                // 分核阈值
    static constexpr int32_t AXES_STEP = 2;                                   // 轴间隔（A/R交替）
    static constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;              // Workspace大小
    static constexpr uint64_t BASIC_BLOCK = 64 * 1024UL;                      // 基础Block大小
    static constexpr uint64_t POST_BUF_SIZE = 8 * 1024UL;                     // Post Reduce输出缓冲
    static constexpr uint64_t CACHE_BUF_SIZE = 16 * 1024UL;                   // Binary Reduce缓存
    static constexpr uint64_t vRegSize = 256;                                 // Vector寄存器大小
    static constexpr auto& memPolicy = BlockOp::memPolicy;

    // ========== Shape变换相关成员 ==========
    std::vector<uint64_t> shapeInfo_;      // 原始输入Shape
    std::vector<uint64_t> axes_;      // Reduce轴索引列表
    uint64_t shape_[MAX_DIM] = {0};     // 变换后的Shape
    uint64_t srcStride_[MAX_DIM] = {0};    // 输入Stride
    uint64_t outStride_[MAX_DIM] = {0};    // 输出Stride
    int32_t useNddma = 0;
    int32_t inputDtypeSize_ = 1;
    uint64_t shapeA_ = 1;
    uint64_t shapeR_ = 1;
    uint64_t reservedNode_ = 0;
    uint64_t reservedSize_ = 0;

    // ========== Tiling计算结果 ==========
    uint64_t basicBlock_;  // UB搬入的buffer大小
    uint64_t resultBlock_; // Reduce计算后的buffer大小
    SliceDescriptor unitA_;  // A轴切分单元
    SliceDescriptor unitR_;  // R轴切分单元
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
    {   // 分开每一步，分不同的模块
        //要加什么校验
        compileInfo_ = Atvoss::GetOpPlatformInfo();
        shapeInfo_ = std::get<0>(std::get<0>(arguments)).shape_vector();
        //size()==0后续有没有对应处理逻辑
        if (shapeInfo_.size() == 0) {
            printf("[ERROR]: [Atvoss][Kernel] Shape info error \n");
            return false;
        }
        // 通过属性名 "axes" 获取 reduce 轴信息
        auto& attrTuple = std::get<1>(arguments);
        auto* axesPtr = FindAttrByName(attrTuple, "axes");
        if (axesPtr == nullptr) {
            printf("[ERROR]: [Atvoss][Kernel] Cannot find 'axes' attribute\n");
            return false;
        }
        axes_ = *axesPtr;
        if(!CheckAxes()){
            return false;
        }
        MakeWrapDim();
        //获取输入size of dtype
        inputDtypeSize_ = 

        if(!DoReduceTiling()){
            return false;
        }
        uint64_t perCoreNum = CeilDiv(unitA_.outer * unitR_.outer, compileInfo_.vectorCoreNum);
        uint64_t numBlocks = CeilDiv(unitA_.outer * unitR_.outer, perCoreNum);
        if (unitA_.outer < numBlocks) {
            auto tmpBlockDim = CeilAlign(numBlocks, unitA_.outer);
            if (tmpBlockDim <= compileInfo_.vectorCoreNum) {
                numBlocks = tmpBlockDim;
            } else {
                numBlocks = FloorAlign(numBlocks, unitA_.outer);
            }
        }

        tilingData.ubFactorA = unitA_.step;
        uint64_t tileACntPerCore = CeilDiv(unitA_.outer, numBlocks);
        tilingData.tileACntPerCore = tileACntPerCore;
        tilingData.totalTileACnt = unitA_.outer;

        tilingData.ubFactorR = unitR_.step;
        uint64_t tileRCntPerCore = CeilDiv(unitR_.outer, CeilDiv(numBlocks, unitA_.outer));
        tilingData.tileRCntPerCore = tileRCntPerCore;
        tilingData.totalTileRCnt = unitR_.outer;
        tilingData.groupR = CeilDiv(unitR_.outer, tileRCntPerCore);
        for (int32_t i = 0; i < MAX_DIM; i++) {
            tilingData.shape[i] = shape_[i];
            tilingData.srcStride[i] = srcStride[i];
            tilingData.outStride[i] = outStride[i];
        }
        uint32_t realCore = CeilDiv(unitA_.outer, tileACntPerCore) * CeilDiv(unitR_.outer, tileRCntPerCore);
        tilingData.coreNum = realCore;
        tilingData.useNddma = useNddma;
        tilingData.resultBlock = resultBlock_;
        tilingData.basicBlock = basicBlock_;
        return true;
    }

private:
    /*!
     * \brief 计算Shape总元素数
     */
    void MakeWrapDim()
    {
        // EnsureNotScalar at least return 1-D Tensor, so shapeSize cannot be 0
        size_t shapeSize = shape_.size();
        for (size_t i = 0; i < axes_.size(); i++) {
            if (axes_[i] < 0) {
                axes_[i] += shapeSize;
            }
        }
        std::sort(axes_.begin(), axes_.end());
    }

    bool CheckAxes()
    {
        int64_t shapeSize = static_cast<int64_t>(shape_.size());
        int64_t axesSize = static_cast<int64_t>(axes_.size());
        //检查axes不能为空
        if (axesSize.size() == 0) {
            printf("illegal axes size:%ld", axesSize);
            return false;
        }
        //axes维度不能超过shape维度
        if (axesSize > shapeSize) {
            printf("illegal axes size:%ld over shape size:%ld", axesSize, shapeSize);
            return false;
        }
        //消去负值后的axes范围必须在[0,shapeSize)范围内
        for (int64_t i = 0; i < axesSize; i++) {
            if (axes_[i] >= shapeSize || axes_[i] < 0) {
                printf("illegal axis:%ld dim:%ld out of shape range:[0, %ld)", i, axes[i], shapeSize);
                return false;
            }
        }
        return true;
    }
    /*!
     * \brief 执行Reduce Tiling计算流程
     *
     * 调用TransformShape进行Shape变换，然后根据维度选择Pattern进行Tiling计算。
     */
    bool DoReduceTiling()
    {
        int32_t newShapeSize = 0;
        TransformShape(newShapeSize);

        if(!DoTilingMatchPattern(newShapeSize)){
            return false;
        }

        ComputeStride();
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
    bool DoTilingMatchPattern(int32_t shapeSize)
    {
        switch (shapeSize) {
            case CONST1:
                return ComputeTiling<__reducePattern::A>();
            case CONST2:
                return ComputeTiling<__reducePattern::AR>();
            case CONST3:
                return ComputeTiling<__reducePattern::ARA>();
            case CONST4:
                return ComputeTiling<__reducePattern::ARAR>();
            case CONST5:
                PadDimOne<__reducePattern::ARARA>();
                return ComputeTiling<__reducePattern::ARARARARA>();
            case CONST6:
                PadDimOne<__reducePattern::ARARAR>();
                return ComputeTiling<__reducePattern::ARARARAR>();
            case CONST7:
                PadDimOne<__reducePattern::ARARARA>();
                return ComputeTiling<__reducePattern::ARARARARA>();
            case CONST8:
                return ComputeTiling<__reducePattern::ARARARAR>();
            case CONST9:
                return ComputeTiling<__reducePattern::ARARARARA>();
            default:
                printf("[ERROR]: [Atvoss][Kernel] Unsupported pattern\n");
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
    void ComputeEmptyTiling()
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
        uint64_t ubAvilSize = compileInfo_.ubSize - CACHE_BUF_SIZE;
        basicBlock_ = FloorAlign(ubAvilSize / CONST2, vRegSize); // double buffer
        for (int32_t i = 1; i < MAX_DIM; i++) {
            shape_[i] = 0;
        }
        shape_[0] = outSize;
        // 空tensor去除R轴后，作为全A的pattern计算切分
        ComputeCacheLineBlock<__reducePattern::A>();
        unitA_.outer *= cBlock_.cacheLineOuter;
        ComputeUnitA<__reducePattern::A>();
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
    bool ComputeTiling()
    {
        // Tiling计算分5步：
        // 1. 根据计算图计算搬入的BasicBlock大小
        // 2. 根据cacheline大小计算cacheline的切分轴，保证搬运性能
        // 3. 计算UB内A轴的切分大小，最大512B或者按A轴分核低于85%
        // 4. 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
        // 5. 可选，针对R轴过小，UB内R轴全载，BasicBlock不能满载是，调整UB内A轴切分，上限为UB内二分缓存大小

        if (IsEmtpyTensor<Pattern>()) {
            if (memPolicy == Atvoss::MemMngPolicy::MANUAL) {
                return CheckAndCalEmptyTiling<Pattern>(TILEA, TILER);
            } else {
                ComputeEmptyTiling<Pattern>();
                return true;
            }
        }

        CalcBasicBlock<Pattern>();
        //空tensor放到calcBasicBlock之前
        if (memPolicy == Atvoss::MemMngPolicy::MANUAL) {
            if (CheckTileShape<Pattern>(TILEA, TILER)) {
                SetSliceDescriptorWithTileShapeA<Pattern>(TILEA);
                if (Pattern::ID != PATTERN_A) {
                    SetSliceDescriptorWithTileShapeR<Pattern>(TILER);
                }
                printf(
                    "tiling step outerA:%lu, innerA:%lu, stepA:%lu, idxA:%d, outerR:%lu, innerR:%lu, stepR:%lu, idxR:%d",
                    unitA_.outer, unitA_.inner, unitA_.step, unitA_.idx, unitR_.outer, unitR_.inner, unitR_.step, unitR_.idx);
                printf("[INFO]: [Atvoss][Kernel] ComputeTiling for pattern ID: %u, Dim: %d\n", Pattern::ID, Pattern::Dim);
                return true;
            } else {
                return false;
            }
        } else {
            ComputeCacheLineBlockAndUnit<Pattern>();

            ComputeUnitA<Pattern>();

            ComputeUnitR<Pattern>();

            ComputeProgressUnitA<Pattern>();

            useNddma = IsUseNddma<Pattern>();

            printf(
                "tiling step outerA:%lu, innerA:%lu, stepA:%lu, idxA:%d, outerR:%lu, innerR:%lu, stepR:%lu, idxR:%d",
                unitA_.outer, unitA_.inner, unitA_.step, unitA_.idx, unitR_.outer, unitR_.inner, unitR_.step, unitR_.idx);
            printf("[INFO]: [Atvoss][Kernel] ComputeTiling for pattern ID: %u, Dim: %d\n", Pattern::ID, Pattern::Dim);

            return true;
        }
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
        uint64_t ubAvilSize = compileInfo_.ubSize - (CACHE_BUF_SIZE + reservedSize_);
        uint64_t ratio = Ratio();

        double basicBlock =
            static_cast<double>(ubAvilSize) / (static_cast<double>(preInputBufferNum + preRestBufferNum * ratio) +
                                            static_cast<double>(postBufferNum * ratio) / static_cast<double>(r));
        return static_cast<uint64_t>(static_cast<double>(basicBlock * ratio) / static_cast<double>(r));
    }

    template <class Pattern>
    uint64_t CaculateReduceSize()
    {
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
        int32_t dim = Pattern::TailA ? Pattern::Dim - AXES_STEP : Pattern::Dim - CONST1;
        uint64_t r = 1;
        for (int32_t i = dim; i > -1; i = i - AXES_STEP) {
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
        if (opDag_.reduceOpPos == 1) {
            // 对于reduce前没有计算的算子，需要额外的inputNum确保搬运流水
            preInputBufferNum = preInputBufferNum * CONST2;
        }
        // reduce后的存活节点加1，用于存放reduce的输出
        uint64_t postBufferNum = ReducedMTE2Num + ReducedTempBufNum + ReducedMTE3Num;

        uint64_t ratio = Ratio();
        uint64_t ubAvilSize = compileInfo_.ubSize - (CACHE_BUF_SIZE + reservedSize_);
        if (Pattern::ID == PATTERN_A) {
            basicBlock_ = ubAvilSize / (preInputBufferNum + (preRestBufferNum + postBufferNum) * ratio);
            basicBlock_ = FloorAlign(basicBlock_, compileInfo_->vRegSize);
            resultBlock_ = basicBlock_ * ratio;
        } else {
            uint64_t resBlock = TryGetReduceBlock<Pattern>(shape_, preInputBufferNum, preRestBufferNum, postBufferNum);
            resultBlock_ = FloorAlign(resBlock, compileInfo_.cacheLineSize);
            if (resultBlock_ < static_cast<uint64_t>(MAX_INNER_A)) {
                resultBlock_ = MAX_INNER_A;
            } else if (resultBlock_ > CACHE_BUF_SIZE) {
                // 输出大小不能超过cache缓存大小, 否则计算额外A轴搬入时会越界
                resultBlock_ = CACHE_BUF_SIZE;
            }
            if (ubAvilSize <= resultBlock_ * postBufferNum) {
                printf(
                    "ubSize:%lu is smaller than size:%lu, not support.", compileInfo_.ubSize,
                    CACHE_BUF_SIZE + reservedSize_ + resultBlock_ * postBufferNum);
            }
            uint64_t preBufSize = ubAvilSize - resultBlock_ * postBufferNum;
            // 按输入和计算大小的比例折算成输入大小
            basicBlock_ = preBufSize / (preInputBufferNum + preRestBufferNum * ratio);
            basicBlock_ = FloorAlign(basicBlock_, vRegSize);
        }
        if (basicBlock_ < vRegSize) {
            printf("basic block:%lu is too small, not support.", basicBlock_);
        }
        printf(
            "preInputBufferNum:%lu, preRestBufferNum:%lu, postBufferNum:%lu, basicBlock:%ld, resBlock:%ld",
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
            printf("tileA exceeds the maximum range of A in the shape. \n");
            return false;
        }
        if (!IsTileMultipleOfLowerDims(AInfo, tileA)) {
            printf("tileA is invalid; it must be a multiple of the lower dimension. \n");
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
            tailABlockAlign = CeilAlign(tailA, compileInfo_.ubBlockSize / MinInputDTypeSize);
            tileABlockAlign =
                tileA < shape_[Pattern::Dim - 1] ? tailABlockAlign : tileA / shape_[Pattern::Dim - 1] * tailABlockAlign;
        } else {
            tailR = tileR < shape_[Pattern::Dim - 1] ? tileR : shape_[Pattern::Dim - 1];
            tailRBlockAlign = CeilAlign(tailR, compileInfo_.ubBlockSize / inputDtypeSize_);
            tileRBlockAlign =
                tileR < shape_[Pattern::Dim - 1] ? tailRBlockAlign : tileR / shape_[Pattern::Dim - 1] * tailRBlockAlign;
        }
        // 检查tileshape是否超过basicblock大小
        if (tileABlockAlign * tileRBlockAlign > basicBlock_) {
            printf("Current tile size exceeds memory limit, please reduce the tile size. \n");
            return false;
        }
        // 检查tileA是否超过A轴缓存最大值
        if (tileABlockAlign > maxInnerA) {
            printf("tileA exceeds the maximum cache size allowed of axis A. \n");
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
            printf("tileA exceeds the maximum range of A in the shape. \n");
            return false;
        }
        if (Pattern::ID != PATTERN_A && tileR > RInfo[RInfo.size() - 1]) {
            printf("tileR exceeds the maximum range of R in the shape. \n");
            return false;
        }
        if (!IsTileMultipleOfLowerDims(AInfo, tileA)) {
            printf("tileA is invalid; it must be a multiple of the lower dimension. \n");
            return false;
        }
        if (Pattern::ID != PATTERN_A && !IsTileMultipleOfLowerDims(RInfo, tileR)) {
            printf("tileR is invalid; it must be a multiple of the lower dimension. \n");
            return false;
        }
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
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
        for (iA = Pattern::TailA ? Pattern::Dim - 1 : Pattern::Dim - 2; iA > -1; iA -= AXES_STEP) {
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
        AssembleUnit(unitA_, iA, innerA, outerA, step);
    }

    template <class Pattern>
    void SetSliceDescriptorWithTileShapeR(uint64_t tileR)
    {
        int32_t iR;
        uint64_t outerR = shapeR_;
        uint64_t innerR = 1;
        uint64_t step = 1;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
        for (iR = Pattern::TailA ? Pattern::Dim - 2 : Pattern::Dim - 1; iR > -1; iR -= AXES_STEP) {
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
        AssembleUnit(unitR_, iR, innerR, outerR, step);
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
        uint64_t cacheSize = compileInfo_.cacheLineSize / inputDtypeSize_;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
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

        cBlock_.cacheLineStep = cacheLineStep;
        cBlock_.cacheLineOuter = cacheLineOuter;
        cBlock_.aSize = aInCacheLine;
        cBlock_.rSize = rInCacheLine;
        printf(
            "cacheLine Block axis:%d, cacheLineStep:%lu, cacheLineOuter:%lu, aSize:%lu, rSize:%lu", cBlock_.axis,
            cBlock_.cacheLineStep, cBlock_.cacheLineOuter, cBlock_.aSize, cBlock_.rSize);
    }

    template <class Pattern>
    void InitUnit()
    {
        int32_t axisInCacheLine = cBlock_.axis;
        for (int32_t i = Pattern::FirstA ? 0 : 1; i < axisInCacheLine; i += AXES_STEP) {
            unitA_.outer *= shape_[i];
        }
        for (int32_t i = Pattern::FirstA ? 1 : 0; i < axisInCacheLine; i += AXES_STEP) {
            unitR_.outer *= shape_[i];
        }
        bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
        if (basicSplitA) {
            unitA_.outer *= cBlock_.cacheLineOuter;
        } else {
            unitR_.outer *= cBlock_.cacheLineOuter;
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
        uint64_t outerA = (basicSplitA ? unitA_.outer / cBlock_.cacheLineOuter * shape_[axisInCacheLine] : unitA_.outer);
        uint64_t innerA = unitA_.inner;
        uint64_t maxCacheA = MAX_INNER_A / MaxInputDTypeSize;
        uint64_t maxInnerA = Pattern::ID == PATTERN_A ? basicBlock_ * Ratio() / MaxInputDTypeSize : maxCacheA;
        uint64_t bBlockNum = basicBlock_ * Ratio() / MaxInputDTypeSize;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
        uint64_t step = 1;
        int32_t iA;
        int32_t lastAxisA = Pattern::TailA ? (Pattern::Dim - 1) : (Pattern::Dim - 2);
        //寻找切轴，什么原则找切分轴
        for (iA = basicSplitA ? axisInCacheLine : axisInCacheLine - 1; iA > -1; iA -= AXES_STEP) {
            uint64_t axisLen = shape_[iA];
            uint64_t stepLen = 1;
            uint64_t innerAOrigin = calOriginInnerA<Pattern>(iA);
            // 根据不同场景，初始化step和步长
            InitStepAndSteplen(&step, &stepLen, iA, lastAxisA, axisInCacheLine, innerAOrigin);
            uint64_t maxStep = step;
            bool splitHere = false;
            //该根轴上寻找合适的ubfactorA
            //RA场景，满足基本分核数的情况下多进A；AR场景，满足基本分核数就结束
            for (; step <= axisLen; step = step + stepLen) {
                uint64_t s = step;
                uint64_t tmpInnerA = innerA * s;
                uint64_t tmpOuterA = outerA / axisLen * CeilDiv(shape_[iA], s);
                uint64_t aSize = tmpInnerA * cBlock_.aSize;
                if (aSize <= maxInnerA && aSize * cBlock_.rSize <= bBlockNum) {
                    uint64_t tempCoreNum = tmpOuterA / CeilDiv(tmpOuterA, compileInfo_.vectorCoreNum);
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
            if (splitHere || maxStep != axisLen || iA - AXES_STEP < 0) {
                // A轴最大、分核最大或者无法继续迭代
                step = maxStep;
                innerA = innerA * step;
                outerA = outerA / axisLen * CeilDiv(shape_[iA], step);
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            outerA /= axisLen;
        }
        AssembleUnit(unitA_, iA, innerA, outerA, step);
    }

    /*
     * 如果当前处理是cacheline切分轴：
     * 如果当前处理最后一根A轴，步长为cacheSize，初始step为cachelinestep和cachesize的最大值；
     * 如果当前处理不是最后一根A轴，根据cacheline切分内A轴乘积是奇数还是偶数分开step和步长为奇数还是偶数。
     * 如果当前处理不是cacheline切分轴：
     * 如果当前处理最后一根A轴，初始step和步长为cacheSize；
     * 如果当前处理不是最后一根A轴，根据cacheline切分内A轴乘积是奇数还是偶数分开step和步长为1还是2。
     */
    void InitStepAndSteplen(uint64_t* Astep, uint64_t* AstepLen, int32_t iA, int32_t lastAxisA, int32_t axisInCacheLine, uint64_t innerAOrigin)
    {
        uint64_t cacheSize = compileInfo_.cacheLineSize / inputDtypeSize_;
        bool isLastAxisA = (iA == lastAxisA);
        //判断cacheline切分内A轴乘积是奇数还是偶数
        bool parityFactor = (cBlock_.aSize != 1 && innerAOrigin % 2 == 1);
        //判断是否是cacheline切分轴
        if (iA == axisInCacheLine) {
            if (isLastAxisA) {
                *Astep = cBlock_.cacheLineStep < cacheSize ? cacheSize : cBlock_.cacheLineStep;
                *AstepLen = cacheSize;
            } else {
                *Astep = parityFactor ? CeilAlign(cBlock_.cacheLineStep, static_cast<uint64_t>(2)) : cBlock_.cacheLineStep;
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
        uint64_t outerR = (basicSplitA ? unitR_.outer : unitR_.outer / cBlock_.cacheLineOuter * shape_[axisInCacheLine]);
        uint64_t innerR = unitR_.inner;
        uint64_t outerA = unitA_.outer;
        uint64_t innerA = unitA_.inner;
        uint64_t step = 1UL;
        uint64_t ubFactorR = 1UL;
        uint64_t bBlockNum = basicBlock_ * Ratio() / MaxInputDTypeSize;
        uint64_t cacheSize = compileInfo_.cacheLineSize / inputDtypeSize_;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
        int32_t iR;
        //找切分轴，原则：找到ub用满的轴
        for (iR = basicSplitA ? axisInCacheLine - 1 : axisInCacheLine; iR > -1; iR -= AXES_STEP) {
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
            uint64_t minStep = (iR == axisInCacheLine ? cBlock_.cacheLineStep : 1UL);
            uint64_t stepLen = (iR == Pattern::Dim - 1) ? cacheSize : 1;
            bool is_prime = isPrime(step);
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
        AssembleUnit(unitR_, iR, innerR, outerR, ubFactorR);
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

            uint64_t tempCoreNum = (outerA * tmpOuterR) / CeilDiv(outerA * tmpOuterR, compileInfo_.vectorCoreNum);
            tempCoreNum = tempCoreNum > outerA ? FloorAlign(tempCoreNum, outerA) : tempCoreNum;
            if (tempCoreNum > THRES_CORENUM) {
                // 找不到合适的rate，就不刷新step
                return tmpS;
            }
        }
        return 1;
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

                uint64_t tempCoreNum = (outerA * tmpOuterR) / CeilDiv(outerA * tmpOuterR, compileInfo_.vectorCoreNum);
                tempCoreNum = tempCoreNum > outerA ? FloorAlign(tempCoreNum, outerA) : tempCoreNum;
                if (tempCoreNum > THRES_CORENUM) {
                    // 找不到合适的rate，就不刷新step
                    return tmpS;
                }
            }
        }
        return 1;
    }
    /*
     * 针对R轴过小，UB内R轴全载，BasicBlock不能满载时，调整UB内A轴切分，上限Reduce计算后输出buffer大小
     */
    template <class Pattern>
    void ComputeProgressUnitA()
    {
        // if RAxis is fully loaded in UB, and basicBlock is not enough, we need to calculate extra unitA
        if (unitR_.idx != -1) {
            return;
        }
        uint64_t innerA = unitA_.inner / unitA_.step; // restore original innerA
        uint64_t outerA = unitA_.outer / CeilDiv(shape_[unitA_.idx], unitA_.step) * shape_[unitA_.idx];
        uint64_t bBlockNum = basicBlock_ * Ratio() / MaxInputDTypeSize;
        uint64_t maxInnerA = resultBlock_ / MaxInputDTypeSize;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
        uint64_t cacheSize = compileInfo_.cacheLineSize / inputDtypeSize_;
        uint64_t innerR = unitR_.inner;
        uint64_t step = 1;
        int32_t iA;
        int32_t lastAxisA = Pattern::TailA ? (Pattern::Dim - 1) : (Pattern::Dim - 2);
        for (iA = unitA_.idx; iA > -1; iA -= AXES_STEP) {
            uint64_t axisLen = shape_[iA];
            uint64_t stepLen = 1;
            uint64_t maxStep = 1;
            InitStepAndSteplenFullLoadR(&step, &stepLen, &maxStep, iA, lastAxisA);
            bool splitHere = false;
            for (; step <= axisLen; step = step + stepLen) {
                uint64_t tmpS = step;
                if (iA == Pattern::Dim - 1) {
                    tmpS = FloorAlign(tmpS, ubBlockSize);
                }
                uint64_t tmpInnerA = innerA * tmpS;
                uint64_t tmpOuterA = outerA / axisLen * CeilDiv(shape_[iA], tmpS);
                uint64_t tempCoreNum = tmpOuterA / CeilDiv(tmpOuterA, compileInfo_.vectorCoreNum);
                tempCoreNum = tempCoreNum > tmpOuterA ? FloorAlign(tempCoreNum, tmpOuterA) : tempCoreNum;
                bool isContinue =
                    (tmpInnerA * innerR * cBlock_.aSize * cBlock_.rSize <= bBlockNum &&
                     tmpInnerA * cBlock_.aSize <= maxInnerA);
                if (isContinue) {
                    if (tmpInnerA * cBlock_.aSize <= cacheSize) {
                        maxStep = tmpS;
                    } else if (tempCoreNum >= THRES_CORENUM) {
                        maxStep = tmpS;
                    }
                    continue;
                } else {
                    splitHere = true;
                    break;
                }
            }
            if (splitHere || iA - AXES_STEP < 0) {
                step = maxStep;
                innerA *= step;
                outerA = outerA / axisLen * CeilDiv(shape_[iA], step);
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            outerA /= axisLen;
        }
        AssembleUnit(unitA_, iA, innerA, outerA, step);
    }

    void InitStepAndSteplenFullLoadR(uint64_t* Astep, uint64_t* AstepLen, uint64_t* maxStepA, int32_t iA, int32_t lastAxisA)
    {
        uint64_t cacheSize = compileInfo_.cacheLineSize / inputDtypeSize_;
        bool isLastAxisA = (iA == lastAxisA);
        //判断是否是cacheline切分轴
        if (iA == unitA_.idx) {
            *Astep = isLastAxisA ? unitA_.step + cacheSize : unitA_.step + 1UL;
            *AstepLen = isLastAxisA ? cacheSize : 1UL;
            *maxStepA = unitA_.step;
        } else {
            *Astep = 1UL;
            *AstepLen = 1UL;
            *maxStepA = 1;
        }
    }

    void AssembleUnit(SliceDescriptor& unit, int32_t idx, uint64_t inner, uint64_t outer, uint64_t step)
    {
        unit.idx = idx;
        unit.inner = inner;
        unit.outer = outer;
        unit.step = step;
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
    void TransformShape(int32_t& shapeSize)
    {
        EliminateOne(shapeSize);
        MergeAxis(shapeSize);
    }

    /*!
     * \brief 消除大小为1且内存连续的维度
     */
    void EliminateOne(int32_t& shapeSize)
    {
        //补充补1原因
        shape_[0] = 1UL;
        int32_t dstIdx = 1; // shape中第一个数给了1, 跳过第一个数
        for (size_t i = 0; i < axes_.size(); i++) {
            // 前面补了一维，所有的axes需要加1
            axes_[i] = axes_[i] + 1;
        }
        int32_t eraseNum = 0;
        for (size_t i = 0; i < shapeInfo_.size(); i++) {
            auto iter = std::find(axes_.begin(), axes_.end(), i + 1);
            if (shapeInfo_[i] != 1L) {
                shape_[dstIdx] = shapeInfo_[i];
                if (iter != reduceAxisInfo.end()) {
                    *iter = *iter - eraseNum;
                }
                dstIdx++;
            } else {
                eraseNum++;
                if (iter != axes_.end()) {
                    axes_.erase(iter);
                }
            }
        }
        shapeSize = dstIdx;
        printf(
            "after EliminateOne, Shape is:%s, axes:%s\n", VectorToString(shape_, shapeSize).c_str(),
            VectorToString(axes_).c_str());
    }

    /*!
     * \brief 合并连续的同类型轴（A轴或R轴）
     *
     * 将连续的多个A轴合并为一个A轴，连续的多个R轴合并为一个R轴，
     * 形成A/R交替的Pattern结构。
     */
    void MergeAxis(int32_t& shapeSize)
    {
        int32_t tmpSize = 0;
        for (int32_t i = 0; i < shapeSize;) {
            auto iter0 = std::find(axes_.begin(), axes_.end(), i);
            bool isRAxis0 = iter0 != axes_.end();
            uint64_t s = shape_[i];
            int32_t j = i + 1;
            for (; j < shapeSize; j++) {
                auto iter1 = std::find(axes_.begin(), axes_.end(), j);
                bool isRAxis1 = iter1 != axes_.end();
                if (isRAxis0 != isRAxis1) {
                    break;
                }
                s *= shape_[j];
                if (isRAxis1) {
                    // 连续的R轴, 需要擦除后续R轴的索引
                    axes_.erase(iter1);
                }
            }
            i = j;
            shape_[tmpSize] = s;
            tmpSize++;
            if (isRAxis0) {
                *iter0 = tmpSize - 1;
            }
        }
        for (int32_t i = tmpSize; i < shapeSize; i++) {
            shape_[i] = 0;
        }
        shapeSize = tmpSize;
        printf(
            "after merge axis, Shape:%s, axes:%s\n", VectorToString(shape_, shapeSize).c_str(),
            VectorToString(axes_).c_str());
    }

    template <class Pattern>
    int32_t IsUseNddma()
    {
        int32_t axis = cBlock_.axis;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / inputDtypeSize_;
        if (viewStride_[Pattern::Dim - 1] != 1UL) {
            // 尾轴非连续, 做NDDMA
            return 1;
        }
        if (shape_[Pattern::Dim - 1] >= ubBlockSize) {
            // last dim 大于ubblock, 不做NDDMA
            return 0;
        }
        if ((Pattern::Dim - 1 - axis > CONST2) || (Pattern::Dim - 1 - axis == CONST2 && cBlock_.cacheLineStep != 1UL)) {
            // cacheline切分超过3维，或者等于三维时，最高维不为1。即cacheline有效切分维度超过2维
            return 1;
        }
        if (Pattern::TailA) {
            uint64_t factorA = unitA_.step;
            for (auto iA = unitA_.idx + AXES_STEP; iA < Pattern::Dim; iA += AXES_STEP) {
                factorA = factorA * shape_[iA];
            }
            if (factorA > ubBlockSize) {
                // 转置后A轴乘积大于ubblock, 不做NDDMA
                return 0;
            }
        } else {
            uint64_t factorR = unitR.step;
            for (auto iR = unitR_.idx + AXES_STEP; iR < Pattern::Dim; iR += AXES_STEP) {
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
            srcStride[dim] = s;
            outStride[dim] = ds;
            s *= shape_[dim];
            if (IsAxisA<Pattern>(dim)) {
                ds *= shape_[dim];
            }
        }
    }

    uint64_t Ratio()
    {
        if (opDag_.memLevel == MemLevel::LEVEL_2) {
            return CeilDiv(MaxInputDTypeSize, static_cast<uint64_t>(inputDtypeSize_));
        }
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
        for (int32_t i = Pattern::TailA ? Pattern::Dim - 1 : Pattern::Dim - 2; i > axis; i -= AXES_STEP) {
            innerAOrigin *= shape_[i];
        }
        return innerAOrigin;
    }

    template <typename T>
    std::string VectorToString(const std::vector<T>& s)
    {
        std::stringstream ss;
        for (auto iter = s.begin(); iter != s.end(); ++iter) {
            ss << *iter;
            if (iter != s.end() - 1) {
                ss << ", ";
            }
        }
        return ss.str();
    }

    template <typename T, typename U>
    std::string VectorToString(const T* s, U size)
    {
        std::stringstream ss;
        for (U i = 0; i < size; i++) {
            ss << s[i];
            if (i != size - 1) {
                ss << ", ";
            }
        }
        return ss.str();
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
    __aicore__ inline BaseKernelSchedule() = default;

    /*!
     * \brief Kernel层执行入口
     *
     * 根据BlockIdx计算当前Core需要处理的数据范围，
     * 构造Block参数，调用Block执行。
     *
     * \param[in] cfg Kernel配置参数（包含tilingData和blockParam）
     * \param[in] args 输入输出GM地址
     */
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        // Configure block scheduling parameters
        typename BlockOp::ScheduleClz::ParamStruct configBlock = cfg.blockParam;
        configBlock.totalElemCnt = CalCurCoreEleCnt(cfg.tilingData);
        // Pack and prepare arguments for kernel execution
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);
        auto params = PrepareParams<Params>(cfg.tilingData, argTuple);
        auto convertArgs = ConvertArgs<Params>(params, argTuple);

        BlockOp blockOp{};
        blockOp.Run(configBlock, convertArgs);
    }

    template <typename Params, typename ArgTup>
    __aicore__ inline auto PrepareParams(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        auto params =
            PrepareParamsImpl<Params>(cfg, argTuple, AscendC::Std::make_index_sequence<Atvoss::Util::Size_v<Params>>{});
        return params;
    }

    /*!
     * \brief 计算当前Core需要处理的元素数量
     */
    __aicore__ inline auto CalCurCoreEleCnt(ScheduleCfg& cfg)
    {
        uint64_t blockNum = cfg.blockNum;
        uint64_t actualNum = cfg.unitNum * cfg.unitNumPerCore;
        if (AscendC::GetBlockIdx() < cfg.moreUnitCoreNum) {
            actualNum += cfg.unitNum;
        }
        if (AscendC::GetBlockIdx() == blockNum - 1) {
            actualNum += cfg.tailNum;
        }
        return actualNum;
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline auto PrepareParamsImpl(ScheduleCfg& config, ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructParam<Atvoss::Util::Get_t<Params, Ints>>(config, args)...);
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline auto ConstructParam(ScheduleCfg& config, ArgTup& args)
    {
        auto arg = AscendC::Std::get<ParamType::number - 1>(args);
        if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
            using DTypeTmp = typename ParamType::Type::PrimType;
            uint64_t offset = CalGMOffset(config);
            auto ptr = (uint64_t)(arg) + sizeof(DTypeTmp) * offset;
            return reinterpret_cast<__gm__ uint8_t*>(ptr);
        } else {
            using argType = decltype(arg);
            if constexpr (std::is_pointer_v<argType>) {
                return *arg;
            } else {
                return arg;
            }
        }
    }

    template <typename Params, size_t Index, typename ParamTup, typename ArgTup>
    __aicore__ inline constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
    {
        constexpr auto pos = Atvoss::Util::Find_v<Atvoss::CheckVarNum<Index + 1>::template Checker, Params>;
        if constexpr (pos < Atvoss::Util::Size_v<Params>) {
            return AscendC::Std::get<pos>(params);
        } else {
            return AscendC::Std::get<Index>(args);
        }
    }

    template <typename Params, typename ParamTup, typename ArgTup, size_t... Ints>
    __aicore__ inline constexpr auto ConvertArgsImpl(
        ParamTup& params, ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConvertOneArg<Params, Ints>(params, args)...);
    }

    template <typename Params, typename ParamTup, typename ArgTup>
    __aicore__ inline auto ConvertArgs(ParamTup& params, ArgTup& args)
    {
        return ConvertArgsImpl<Params>(
            params, args, AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<ArgTup>>{});
    }

    /*!
     * \brief 计算当前Core在GM中的偏移量
     */
    __aicore__ inline auto CalGMOffset(ScheduleCfg& config)
    {
        if (AscendC::GetBlockIdx() < config.moreUnitCoreNum) {
            return AscendC::GetBlockIdx() * (config.unitNumPerCore * config.unitNum + config.unitNum);
        }
        return config.unitNumPerCore * config.unitNum * AscendC::GetBlockIdx() +
               config.moreUnitCoreNum * config.unitNum;
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
