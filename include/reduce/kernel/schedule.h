/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
#define ATVOSS_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
#include <functional>
#include "common/arch.h"
#include "common/compile_info.h"
#include "reduce/reduce_util.h"

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


struct ReduceTilingUnit {
    int32_t idx = -1;   // ub cut axis
    uint64_t inner = 1; // record the cache line aSize's num or rSize's num in ub
    uint64_t outer = 1; // outer size of ub
    uint64_t step = 1;  // step of cacheline
};

struct CacheLineBlock {
    int32_t axis = -1;           // cacheline cut axis
    uint64_t cacheLineStep = 1;  // step len in cacheline axis
    uint64_t cacheLineOuter = 1; // relative to cacheLineStep, out size of cacheline cut axis
    uint64_t aSize = 1;          // A axis size in cacheline
    uint64_t rSize = 1;          // R axis size in cacheline
};

/*!
 * KernelBuilder: Calculate the tiling information, then determine the GM data that the current core needs to process
 * based on the block ID, and pass it to the block to complete the computation.
 */
template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class BaseKernelSchedule {
public:
    using ExprMaker = typename BlockOp::ScheduleClz::ExpressMaker;
    using ParamStruct = ScheduleCfg;
    using BlockTemplate = BlockOp;
    static constexpr auto EXPRESSION = ToLinearizerExpr(ExprMaker{}.template Compute<KernelTensor>());
    using Expr = typename decltype(EXPRESSION)::Type;
    using Params = Atvoss::Params_t<Expr>;
    using TileShape = typename BlockTemplate::BlockTileShape;
    using ArchTag = typename BlockOp::ScheduleClz::ArchTag;
    static Atvoss::OpCompileInfo compileInfo_;

    static constexpr uint64_t TILEA = TileShape::template get_type<0>::value;
    static constexpr uint64_t TILER = TileShape::template get_type<1>::value;
    static constexpr int32_t MAX_INNER_A = 512; // 单位字节
    static constexpr int32_t MEMBOUND_CORENUM = 40;
    static constexpr int32_t VECBOUND_CORENUM = 48;
    static constexpr int32_t THRES_CORENUM = MEMBOUND_CORENUM;
    static constexpr int32_t A_STEP_LEN = 4;
    static constexpr int32_t AXES_STEP = 2;
    static constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024; // fixed workspace size for ascendc
    static constexpr uint64_t BASIC_BLOCK = 64 * 1024UL;
    static constexpr uint64_t POST_BUF_SIZE = 8 * 1024UL;   // post reduce size for ub reduce
    static constexpr uint64_t CACHE_BUF_SIZE = 16 * 1024UL; // cache for binary reduce
    static constexpr uint64_t vRegSize = 256;

    std::vector<uint64_t> shapeInfo;
    std::vector<int> reduceAxisInfo;
    uint64_t newShape[MAX_DIM] = {0};
    uint64_t srcstride[MAX_DIM] = {0};
    uint64_t outStride[MAX_DIM] = {0};
    std::vector<int64_t> dimStrides;
    uint64_t sliceNum_[MAX_DIM] = {0};
    uint64_t sliceShape_[MAX_DIM] = {0};
    uint64_t sliceStride_[MAX_DIM] = {0};
    int64_t viewStride_[MAX_DIM] = {0};
    uint64_t basicBlock_{0};  // 算子搬入的buffer大小
    uint64_t resultBlock_{0}; // reduce计算后的buffer大小
    ReduceTilingUnit unitA_;
    ReduceTilingUnit unitR_;
    CacheLineBlock cBlock_;
    /*!
     * \brief Kernel layer execution function.
     * \param[in] arguments, information of the user inputs.
     * \param[in] kernelParam, Tiling information in kernel.
     * \return bool. Return true to indicate param calculation success.
     */
    template <typename Args>
    static bool MakeScheduleConfig(const Args& arguments, ScheduleCfg& kernelParam)
    {
        // Extract input shape infomation from arguments
        using InputTuple = std::decay_t<decltype(std::get<0>(arguments))>;
        static_assert(std::tuple_size_v<InputTuple> > 0, "[ERROR]: [Atvoss][Kernel] Get input shape error \n");
        compileInfo_ = Atvoss::GetOpCompileInfo();
        shapeInfo = std::get<0>(std::get<0>(arguments)).shape_vector();
        uint64_t totalEleNum = CalculateTotalElements(shapeInfo);
        reduceAxisInfo = std::get<0>(std::get<1>(arguments)).value;
        if (shapeInfo.size() == 0 || totalEleNum == 0) {
            printf("[ERROR]: [Atvoss][Kernel] Shape info error \n");
            return false;
        }
        int32_t shapeSize = shapeInfo.size();
        int64_t stride = 1L;
        dimStrides.resize(shapeSize);
        for (int32_t i = shapeSize - 1; i >= 0; i--) {
            dimStrides[i] = stride;
            stride = stride * shapeInfo[i];
        }

        DoReduceTiling(newShape);
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

        kernelParam.ubFactorA = unitA_.step;
        uint64_t tileACntPerCore = CeilDiv(unitA_.outer, numBlocks);
        kernelParam.tileACntPerCore = tileACntPerCore;
        kernelParam.totalTileACnt = unitA_.outer;

        kernelParam.ubFactorR = unitR_.step;
        uint64_t tileRCntPerCore = CeilDiv(unitR_.outer, CeilDiv(numBlocks, unitA_.outer));
        kernelParam.tileRCntPerCore = tileRCntPerCore;
        kernelParam.totalTileRCnt = unitR_.outer;
        kernelParam.groupR = CeilDiv(unitR_.outer, tileRCntPerCore);
        for (int32_t i = 0; i < MAX_DIM; i++) {
            kernelParam.shape[i] = newShape[i];
            kernelParam.srcStride[i] = srcStride[i];
            kernelParam.outStride[i] = outStride[i];
        }
        uint32_t realCore = CeilDiv(unitA_.outer, tileACntPerCore) * CeilDiv(unitR_.outer, tileRCntPerCore);
        kernelParam.coreNum = realCore;
        return true;
    }

private:
    static inline uint64_t CalculateTotalElements(const std::vector<uint64_t>& shapeInfo)
    {
        uint64_t totalEleNum = 1;
        for (auto dim : shapeInfo) {
            totalEleNum *= dim;
        }
        return totalEleNum;
    }

    static void DoReduceTiling(uint64_t* newShape)
    {
        int64_t newStride[MAX_DIM] = {0};
        int32_t newShapeSize = 0;
        TransformShape(newShape, newStride, newShapeSize);

        DoTilingMatchPattern(newShape, newShapeSize);

        ComputeStride(newShape);
    }

    static void DoTilingMatchPattern(uint64_t* shape, int32_t shapeSize)
    {
        switch (shapeSize) {
            case CONST1:
                ComputeTiling<__reducePattern::A>(shape);
                break;
            case CONST2:
                ComputeTiling<__reducePattern::AR>(shape);
                break;
            case CONST3:
                ComputeTiling<__reducePattern::ARA>(shape);
                break;
            case CONST4:
                ComputeTiling<__reducePattern::ARAR>(shape);
                break;
            case CONST5:
                PadDimOne<__reducePattern::ARARA>(shape);
                ComputeTiling<__reducePattern::ARARARARA>(shape);
                break;
            case CONST6:
                PadDimOne<__reducePattern::ARARAR>(shape);
                ComputeTiling<__reducePattern::ARARARAR>(shape);
                break;
            case CONST7:
                PadDimOne<__reducePattern::ARARARA>(shape);
                ComputeTiling<__reducePattern::ARARARARA>(shape);
                break;
            case CONST8:
                ComputeTiling<__reducePattern::ARARARAR>(shape);
                break;
            case CONST9:
                ComputeTiling<__reducePattern::ARARARARA>(shape);
                break;
            default:
                printf("[ERROR]: [Atvoss][Kernel] Unsupported pattern\n");
                break;
        }
    }

    template <class Pattern>
    static void PadDimOne(uint64_t* shape)
    {
        int32_t padNum = Pattern::TailA ? CONST9 - Pattern::Dim : CONST8 - Pattern::Dim;
        int32_t maxDim = Pattern::TailA ? CONST9 : CONST8;
        if (padNum == 0) {
            return;
        }
        for (int32_t i = 0; i < Pattern::Dim; ++i) {
            shape[maxDim - 1 - i] = shape[Pattern::Dim - 1 - i];
            viewStride_[maxDim - 1 - i] = viewStride_[Pattern::Dim - 1 - i];
            sliceNum_[maxDim - 1 - i] = sliceNum_[Pattern::Dim - 1 - i];
            sliceShape_[maxDim - 1 - i] = sliceShape_[Pattern::Dim - 1 - i];
            sliceStride_[maxDim - 1 - i] = sliceStride_[Pattern::Dim - 1 - i];
        }
        for (int32_t i = 0; i < padNum; ++i) {
            shape[i] = 1UL;
            sliceNum_[i] = 1UL;
            sliceShape_[i] = 1UL;
            viewStride_[i] = viewStride_[padNum] * shape[padNum];
            sliceStride_[i] = sliceStride_[padNum] * sliceNum_[padNum];
        }
    }

    template <class Pattern>
    static void ComputeEmptyTiling(uint64_t* shape)
    {
        uint64_t outSize = 1;
        for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
            if (IsAxisA<Pattern>(dim)) {
                outSize *= shape[dim];
            }
        }
        context_->SetBlockDim(compileInfo_->vectorCoreNum);
        if (outSize == 0) {
            return;
        }

        uint64_t ubAvilSize = compileInfo_->ubSize - CACHE_BUF_SIZE;
        basicBlock_ = FloorAlign(ubAvilSize / CONST2, vRegSize); // double buffer
        for (int32_t i = 1; i < MAX_DIM; i++) {
            shape[i] = 0;
        }
        shape[0] = outSize;
        // 空tensor去除R轴后，作为全A的pattern计算切分
        ComputeCacheLineBlock<__reducePattern::A>(shape);
        unitA_.outer *= cBlock_.cacheLineOuter;
        // 空tensor场景，按A轴合轴的结果重新赋值sliceShape和sliceNum
        sliceShape_[0] = outSize;
        sliceNum_[0] = 1;
        ComputeUnitA<__reducePattern::A>(shape);
        SetTilingData<__reducePattern::A>(shape);
    }

    template <class Pattern>
    static void ComputeTiling(uint64_t* shape)
    {
        // Tiling计算分5步：
        // 1. 根据计算图计算搬入的BasicBlock大小
        // 2. 根据cacheline大小计算cacheline的切分轴，保证搬运性能
        // 3. 计算UB内A轴的切分大小，最大512B或者按A轴分核低于85%
        // 4. 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
        // 5. 可选，针对R轴过小，UB内R轴全载，BasicBlock不能满载是，调整UB内A轴切分，上限为UB内二分缓存大小

        CalcBasicBlock<Pattern>(shape);
        if (IsEmtpyTensor<Pattern>(shape)) {
            ComputeEmptyTiling<Pattern>(shape);
            return;
        }
        CheckTileShape<Pattern>(shape, TILEA, TILER);
        ComputeCacheLineBlockAndUnit<Pattern>(shape);

        ComputeUnitA<Pattern>(shape);

        ComputeUnitR<Pattern>(shape);

        ComputeProgressUnitA<Pattern>(shape);

        printf(
            "tiling step outerA:%lu, innerA:%lu, stepA:%lu, idxA:%d, outerR:%lu, innerR:%lu, stepR:%lu, idxR:%d",
            unitA_.outer, unitA_.inner, unitA_.step, unitA_.idx, unitR_.outer, unitR_.inner, unitR_.step, unitR_.idx);
        printf("[INFO]: [Atvoss][Kernel] ComputeTiling for pattern ID: %u, Dim: %d\n", Pattern::ID, Pattern::Dim);
    }

    // try get reduce result block size
    template <class Pattern>
    uint64_t TryGetReduceBlock(
        const uint64_t* shape, uint64_t preInputBufferNum, uint64_t preRestBufferNum, uint64_t postBufferNum)
    {
        uint64_t r = CaculateReduceSize<Pattern>(shape);
        // basic * (preMTE * 2 + preVec + 1) + (basic / r) * (postMTE + postVec) = ub - reserved
        uint64_t ubAvilSize = compileInfo_.ubSize - (CACHE_BUF_SIZE + opInput_.reservedSize);
        uint64_t ratio = Ratio();

        double basicBlock =
            static_cast<double>(ubAvilSize) / (static_cast<double>(PreMTE * 2 + PreVec + 1) +
                                            static_cast<double>(PostMTE + PostVec) / static_cast<double>(r));
        return static_cast<uint64_t>(static_cast<double>(basicBlock * ratio) / static_cast<double>(r));
    }

    template <class Pattern>
    uint64_t CaculateReduceSize(const uint64_t* shape)
    {
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        int32_t dim = Pattern::TailA ? Pattern::Dim - AXES_STEP : Pattern::Dim - CONST1;
        uint64_t r = 1;
        for (int32_t i = dim; i > -1; i = i - AXES_STEP) {
            if (i == Pattern::Dim - 1) {
                r = r * CeilAlign(shape[i], ubBlockSize);
            } else {
                r = r * shape[i];
            }
        }
        return r;
    }

    template <class Pattern>
    static void CalcBasicBlock(const uint64_t* shape)
    {
        uint64_t PreMTE = 1;
        uint64_t PreVec = 1;
        uint64_t PostMTE = 1;
        uint64_t PostVec = 1;

        uint64_t ratio = Ratio();
        uint64_t ubAvilSize = compileInfo_.ubSize - (CACHE_BUF_SIZE + opInput_.reservedSize);
        if (Pattern::ID == PATTERN_A) {
            basicBlock_ = ubAvilSize / (preInputBufferNum + (preRestBufferNum + postBufferNum) * ratio);
            basicBlock_ = FloorAlign(basicBlock_, vRegSize);
            resultBlock_ = basicBlock_ * ratio;
        } else {
            uint64_t resBlock = TryGetReduceBlock<Pattern>(shape, PreMTE, PreVec, PostMTE, PostVec);
            resultBlock_ = FloorAlign(resBlock, compileInfo_.cacheLineSize);
            if (resultBlock_ < static_cast<uint64_t>(MAX_INNER_A)) {
                resultBlock_ = MAX_INNER_A;
            } else if (resultBlock_ > CACHE_BUF_SIZE) {
                // 输出大小不能超过cache缓存大小, 否则计算额外A轴搬入时会越界
                resultBlock_ = CACHE_BUF_SIZE;
            }
            if (ubAvilSize <= resultBlock_ * postBufferNum) {
                printf("ubSize:%lu is smaller than size:%lu, not support.", compileInfo_.ubSize,
                    CACHE_BUF_SIZE + opInput_.reservedSize + resultBlock_ * postBufferNum);
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

        // CalcUserBasicBlock(Pattern::ID == PATTERN_A);
        return;
    }

    template <class Pattern>
    static void CheckTileShape(uint64_t* shape, uint64_t tileA, uint64_t tileR)
    {
        uint64_t tailABlockAlign = 1;
        uint64_t tailRBlockAlign = 1;
        uint64_t tailA = 1;
        uint64_t tailR = 1;
        uint64_t tileABlockAlign = tileA;
        uint64_t tileRBlockAlign = tileR;
        uint64_t maxCacheA = MAX_INNER_A / opDag_.maxInputBytes;
        uint64_t maxInnerA = Pattern::ID == PATTERN_A ? basicBlock_ * Ratio() / opDag_.maxInputBytes : maxCacheA;
        if (Pattern::TailA) {
            tailA = tileA < shape[Pattern::Dim - 1] ? tileA : shape[Pattern::Dim - 1];
            tailABlockAlign = CeilAlign(tailA, compileInfo_.ubBlockSize / opDag_.minInputBytes);
            tileABlockAlign = tileA < shape[Pattern::Dim - 1] ? tailABlockAlign : tileA / shape[Pattern::Dim - 1] * tailABlockAlign;
        } else {
            tailR = tileR < shape[Pattern::Dim - 1] ? tileR : shape[Pattern::Dim - 1];
            tailRBlockAlign = CeilAlign(tailR, compileInfo_.ubBlockSize / ge::GetSizeByDataType(opInput_.inputDtype));
            tileRBlockAlign = tileR < shape[Pattern::Dim - 1] ? tailRBlockAlign : tileR / shape[Pattern::Dim - 1] * tailRBlockAlign;
        }
        // 检查tileshape是否超过basicblock大小
        if (tileABlockAlign * tileRBlockAlign > basicBlock_) {
            printf("Current tile size exceeds memory limit, please reduce the tile size.");
            return;
        }
        //检查tileA是否超过A轴缓存最大值
        if (tileABlockAlign > maxInnerA) {
            printf("tileA exceeds the maximum cache size allowed of axis A.");
            return;
        }
        //检查tileA和tileR是否满足低维轴的整数倍
        std::vector<uint64_t> AInfo;
        std::vector<uint64_t> RInfo;
        uint64_t baseA = 1;
        uint64_t baseR = 1;
        for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
            if (IsAxisA<Pattern>(dim)) {
                baseA *= shape[dim];
                AInfo.push_back(baseA);
            } else {
                baseR *= shape[dim];
                RInfo.push_back(baseR);
            }
        }
        if (tileA > AInfo[AInfo.size()-1]){
            printf("tileA exceeds the maximum range of A in the shape");
            return;
        }
        if (Pattern::ID != PATTERN_A && tileR > RInfo[RInfo.size()-1]){
            printf("tileR exceeds the maximum range of R in the shape");
            return;
        }
        if (!IsTileMultipleOfLowerDims(AInfo, tileA)){
            printf("tileA is invalid; it must be a multiple of the lower dimension.");
            return;
        }
        if (Pattern::ID != PATTERN_A && !IsTileMultipleOfLowerDims(RInfo, tileR)){
            printf("tileR is invalid; it must be a multiple of the lower dimension.");
            return;
        }
        CalUserDefineTilingA<Pattern>(shape, AInfo[AInfo.size()-1], tileA);
        if (Pattern::ID != PATTERN_A) {
            CalUserDefineTilingR<Pattern>(shape, RInfo[RInfo.size()-1], tileR);
        }
    }

    static bool IsTileMultipleOfLowerDims(std::vector<uint64_t> shapeInfo, uint64_t tile)
    {
        uint64_t idx = 0;
        for (uint64_t i=0; i<shapeInfo.size(); i++) {
            if (tile <= shapeInfo[i]) {
                idx = i;
                break;
            }
        }
        if (idx != 0 && tile % shapeInfo[idx - 1]!=0) {
            return false;
        }
        return true;
    }

    template <class Pattern>
    static void CalUserDefineTilingA(uint64_t* shape, uint64_t shapeA, uint64_t tileA)
    {
        int32_t iA;
        uint64_t outerA = shapeA;
        uint64_t innerA = 1;
        uint64_t step = 1;
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        for (iA = Pattern::TailA ? Pattern::Dim - 1 : Pattern::Dim - 2; iA > -1; iA -= AXES_STEP) {
            uint64_t axisLen = shape[iA];
            outerA /= axisLen;
            if (tileA <= axisLen) {
                innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(tileA, ubBlockSize) : tileA);
                outerA *= CeilDiv(axisLen, tileA);
                step =  tileA;
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            tileA /= axisLen;
        }
        AssembleUnit(unitA_, iA, innerA, outerA, step);
    }

    template <class Pattern>
    static void CalUserDefineTilingR(uint64_t* shape, uint64_t shapeR, uint64_t tileR)
    {
        int32_t iR;
        uint64_t outerR = shapeR;
        uint64_t innerR = 1;
        uint64_t step = 1;
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        for (iR = Pattern::TailA ? Pattern::Dim - 2 : Pattern::Dim - 1; iR > -1; iR -= AXES_STEP) {
            uint64_t axisLen = shape[iR];
            outerR /= axisLen;
            if (tileR <= axisLen) {
                innerR *= (iR == Pattern::Dim - 1 ? CeilAlign(tileR, ubBlockSize) : tileR);
                outerR *= CeilDiv(axisLen, tileR);
                step =  tileR;
                break;
            }
            innerR *= (iR == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            tileR = tileR / axisLen;
        }
        AssembleUnit(unitR_, iR, innerR, outerR, step);
    }

    template <class Pattern>
    static bool IsEmtpyTensor(const uint64_t* shape)
    {
        for (int32_t i = 0; i < Pattern::Dim; i++) {
            if (shape[i] == 0) {
                return true;
            }
        }
        return false;
    }

    template <class Pattern>
    static void ComputeCacheLineBlock(const uint64_t* shape)
    {
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t cacheSize = compileInfo_.cacheLineSize / dSize;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        uint64_t cacheLineShape = 1UL;
        uint64_t cacheLineStep = 1UL;
        uint64_t cacheLineOuter = 1UL;
        uint64_t aInCacheLine = 1UL;
        uint64_t rInCacheLine = 1UL;

        for (int32_t i = Pattern::Dim - 1; i > -1; --i) {
            cacheLineShape *= shape[i];
            if (cacheLineShape > cacheSize) {
                cacheLineShape /= shape[i];
                cacheLineStep = CeilDiv(cacheSize, cacheLineShape); // sliceshape
                // 非连续与sliceShape对齐
                cacheLineStep = cacheLineStep > sliceShape_[i] ? FloorAlign(cacheLineStep, sliceShape_[i]) : cacheLineStep;
                if (cacheLineStep >= sliceShape_[i]) {
                    cacheLineOuter = CeilDiv(shape[i], cacheLineStep);
                } else {
                    cacheLineOuter = sliceNum_[i] * CeilDiv(sliceShape_[i], cacheLineStep);
                }
                cBlock_.axis = i;
                break;
            } else {
                cacheLineStep = shape[i];
                cBlock_.axis = i;
            }
        }

        for (int32_t i = Pattern::Dim - 1; i > cBlock_.axis; --i) {
            if (i == Pattern::Dim - 1) {
                if (IsAxisA<Pattern>(i)) {
                    aInCacheLine = aInCacheLine * CeilAlign(shape[i], ubBlockSize);
                } else {
                    rInCacheLine = rInCacheLine * CeilAlign(shape[i], ubBlockSize);
                }
            } else {
                if (IsAxisA<Pattern>(i)) {
                    aInCacheLine = aInCacheLine * shape[i];
                } else {
                    rInCacheLine = rInCacheLine * shape[i];
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
    static void InitUnit(const uint64_t* shape)
    {
        int32_t axisInCacheLine = cBlock_.axis;
        for (int32_t i = Pattern::FirstA ? 0 : 1; i < axisInCacheLine; i += AXES_STEP) {
            unitA_.outer *= shape[i];
        }
        for (int32_t i = Pattern::FirstA ? 1 : 0; i < axisInCacheLine; i += AXES_STEP) {
            unitR_.outer *= shape[i];
        }
        bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
        if (basicSplitA) {
            unitA_.outer *= cBlock_.cacheLineOuter;
        } else {
            unitR_.outer *= cBlock_.cacheLineOuter;
        }
    }

    template <class Pattern>
    static void ComputeCacheLineBlockAndUnit(const uint64_t* shape)
    {
        ComputeCacheLineBlock<Pattern>(shape);
        InitUnit<Pattern>(shape);
    }

    /*
    * 计算UB内A轴的切分大小，最大512B或者按A轴分核低于85%
    * 例如: float32, shape:(12800, 8), pattern:AR, cacheLine:256B, coreNum:64
    * cacheLine切分后(1600, cacheLine(8, 8))
    * 对1600做A轴切分，找到uintA inner:16(cacheline中A轴为8, 与16相乘后达到512B上限), outer:100
    */
    template <class Pattern>
    static void ComputeUnitA(const uint64_t* shape)
    {
        int32_t axisInCacheLine = cBlock_.axis;
        bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
        uint64_t outerA = (basicSplitA ? unitA_.outer / cBlock_.cacheLineOuter * shape[axisInCacheLine] : unitA_.outer);
        uint64_t innerA = unitA_.inner;
        uint64_t maxCacheA = MAX_INNER_A / opDag_.maxInputBytes;
        uint64_t maxInnerA = Pattern::ID == PATTERN_A ? basicBlock_ * Ratio() / opDag_.maxInputBytes : maxCacheA;
        uint64_t stepLenInit = Pattern::ID == PATTERN_A ? A_STEP_LEN : 1; // 纯A的步长为4, 减少循环次数
        uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        uint64_t cacheSize = compileInfo_.cacheLineSize / dSize;
        uint64_t totalRSize = CaculateReduceSize<Pattern>(shape);
        //判断R轴是否可以全载，如果可以全载，按照全载R切分A
        uint64_t aInCacheLineSize = basicSplitA ? cBlock_.aSize * cBlock_.cacheLineStep : cBlock_.aSize;
        bool rFullLoad = (bBlockNum / totalRSize >= aInCacheLineSize);
        uint64_t step = 1;
        int32_t iA;
        int32_t useNddma = IsUseNddma<Pattern>(shape);
        for (iA = basicSplitA ? axisInCacheLine : axisInCacheLine - 1; iA > -1; iA -= AXES_STEP) {
            uint64_t axisLen = shape[iA];
            // cacheline切分轴从cacheLineStep开始找，非cacheline切分轴从1开始
            step = (iA == axisInCacheLine ? cBlock_.cacheLineStep : 1UL);
            uint64_t stepLen = (iA == Pattern::Dim - 1 && Pattern::ID != PATTERN_A) ? cacheSize : stepLenInit;
            uint64_t maxStep = (iA == axisInCacheLine ? cBlock_.cacheLineStep : 1UL);
            bool splitHere = false;
            uint64_t maxCoreNum = 0;
            for (; step <= axisLen; step = step + stepLen) {
                uint64_t s = step > sliceShape_[iA] ? FloorAlign(step, sliceShape_[iA]) : step; // 非连续与sliceShape对齐
                if (iA == Pattern::Dim - 1 && s != sliceShape_[iA]) {
                    s = FloorAlign(s, ubBlockSize);
                }
                uint64_t tmpInnerA = innerA * s;
                uint64_t tmpOuterA =
                    (s >= sliceShape_[iA] ? outerA / axisLen * CeilDiv(axisLen, s) :
                                            outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], s));
                uint64_t aSize = tmpInnerA * cBlock_.aSize;
                //R能全载的情况下，按照全载R切分A轴
                if (bBlockNum /aSize < totalRSize && rFullLoad) {
                    splitHere = true;
                    break;
                }
                if (aSize <= maxInnerA && aSize * cBlock_.rSize <= bBlockNum) {
                    uint64_t tempCoreNum =
                        tmpOuterA / CeilDiv(tmpOuterA, compileInfo_.vectorCoreNum);
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
                    maxCoreNum = tempCoreNum > maxCoreNum ? tempCoreNum : maxCoreNum;
                } else {
                    splitHere = true;
                    break;
                }
            }
            if (splitHere || maxStep != axisLen || iA - AXES_STEP < 0) {
                // A轴最大、分核最大或者无法继续迭代
                step = maxStep == 0 ? 1UL : maxStep;
                if (useNddma == 1 && axisLen != 1) {
                    step = CeilAlign(step, static_cast<uint64_t>(CONST2));
                }
                innerA = innerA * step;
                outerA =
                    (step >= sliceShape_[iA] ? outerA / axisLen * CeilDiv(axisLen, step) :
                                            outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], step));
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            outerA /= axisLen;
        }
        AssembleUnit(unitA_, iA, innerA, outerA, step);
    }

    /*
    * 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
    * 用BasicBlock / UB内A的切分大小，找到满BasicBlock的R轴切分位置
    */
    template <class Pattern>
    static void ComputeUnitR(const uint64_t* shape)
    {
        int32_t axisInCacheLine = cBlock_.axis;
        bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
        uint64_t outerR = (basicSplitA ? unitR_.outer : unitR_.outer / cBlock_.cacheLineOuter * shape[axisInCacheLine]);
        uint64_t innerR = unitR_.inner;
        uint64_t outerA = unitA_.outer;
        uint64_t innerA = unitA_.inner;
        uint64_t step = 1UL;
        uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t cacheSize = compileInfo_.cacheLineSize / dSize;
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        int32_t iR;
        for (iR = basicSplitA ? axisInCacheLine - 1 : axisInCacheLine; iR > -1; iR -= AXES_STEP) {
            uint64_t axisLen = shape[iR];
            if (innerA * innerR * cBlock_.aSize * cBlock_.rSize * axisLen <= bBlockNum) {
                outerR = outerR / axisLen;
                innerR *= (iR == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
                continue;
            }

            // maybe bBlockNum not full
            step = std::min(bBlockNum / (innerA * innerR * cBlock_.aSize * cBlock_.rSize), axisLen);
            if (step > sliceShape_[iR]) {
                step = FloorAlign(step, sliceShape_[iR]);
            }
            if (iR == Pattern::Dim - 1) {
                step = FloorAlign(step, ubBlockSize);
            }
            uint64_t minStep = (iR == axisInCacheLine ? cBlock_.cacheLineStep : 1UL);
            uint64_t s = step;
            uint64_t stepLen = (iR == Pattern::Dim - 1) ? cacheSize : 1;
            for (; s > minStep; s = s - stepLen) {
                uint64_t tmpS = (s > sliceShape_[iR] ? FloorAlign(s, sliceShape_[iR]) : s); // 非连续与sliceShape对齐
                if (iR == Pattern::Dim - 1 && tmpS != sliceShape_[iR]) {
                    // 尾轴R场景，尽可能保证尾轴切分block对齐，减少kernel侧补pad
                    tmpS = FloorAlign(tmpS, ubBlockSize);
                }
                auto tmpOuterR =
                    (tmpS >= sliceShape_[iR] ? outerR / axisLen * CeilDiv(axisLen, tmpS) :
                                            outerR / axisLen * sliceNum_[iR] * CeilDiv(sliceShape_[iR], tmpS));

                uint64_t tempCoreNum = (outerA * tmpOuterR) / CeilDiv(outerA * tmpOuterR, compileInfo_.vectorCoreNum);
                tempCoreNum = tempCoreNum > outerA ? FloorAlign(tempCoreNum, outerA) : tempCoreNum;
                if (tempCoreNum > THRES_CORENUM) {
                    // 找不到合适的rate，就不刷新step
                    step = tmpS;
                    break;
                }
            }
            //R轴非尾轴时，继续遍历寻找能够均分切分轴的R
            if (iR != Pattern::Dim - 1) {
                for (; s > minStep; s--) {
                    if (axisLen % s == 0) {
                        uint64_t tmpS = s;
                        auto tmpOuterR =
                            (tmpS >= sliceShape_[iR] ? outerR / axisLen * CeilDiv(axisLen, tmpS) :
                                                    outerR / axisLen * sliceNum_[iR] * CeilDiv(sliceShape_[iR], tmpS));

                        uint64_t tempCoreNum = (outerA * tmpOuterR) / CeilDiv(outerA * tmpOuterR, compileInfo_.vectorCoreNum);
                        tempCoreNum = tempCoreNum > outerA ? FloorAlign(tempCoreNum, outerA) : tempCoreNum;
                        if (tempCoreNum > THRES_CORENUM) {
                            // 找不到合适的rate，就不刷新step
                            step = tmpS;
                            break;
                        }
                    }
                }
            }
            innerR = innerR * step;
            outerR =
                (step >= sliceShape_[iR] ? outerR / axisLen * CeilDiv(axisLen, step) :
                                        outerR / axisLen * sliceNum_[iR] * CeilDiv(sliceShape_[iR], step));
            break;
        }
        AssembleUnit(unitR_, iR, innerR, outerR, step);
    }

    /*
    * 针对R轴过小，UB内R轴全载，BasicBlock不能满载时，调整UB内A轴切分，上限Reduce计算后输出buffer大小
    */
    template <class Pattern>
    static void ComputeProgressUnitA(const uint64_t* shape)
    {
        // if RAxis is fully loaded in UB, and basicBlock is not enough, we need to calculate extra unitA
        if (unitR_.idx != -1) {
            return;
        }
        uint64_t innerA = unitA_.inner / unitA_.step; // restore original innerA
        uint64_t outerA = unitA_.outer / CeilDiv(shape[unitA_.idx], unitA_.step) * shape[unitA_.idx];
        if (unitA_.step < sliceShape_[unitA_.idx]) {
            outerA =
                unitA_.outer / sliceNum_[unitA_.idx] / CeilDiv(sliceShape_[unitA_.idx], unitA_.step) * shape[unitA_.idx];
        }
        uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
        uint64_t maxInnerA = resultBlock_ / opDag_.maxInputBytes;
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        uint64_t cacheSize = compileInfo_.cacheLineSize / dSize;
        uint64_t innerR = unitR_.inner;
        uint64_t step = 1;
        int32_t iA;
        for (iA = unitA_.idx; iA > -1; iA -= AXES_STEP) {
            uint64_t axisLen = shape[iA];
            bool splitHere = false;
            uint64_t s = (iA == unitA_.idx ? unitA_.step + 1UL : 1UL);
            uint64_t maxStep = (iA == unitA_.idx ? unitA_.step : 1UL);
            for (; s <= axisLen; s++) {
                uint64_t tmpS = s > sliceShape_[iA] ? FloorAlign(s, sliceShape_[iA]) : s; // 非连续与sliceShape对齐
                if (iA == Pattern::Dim - 1 && tmpS != sliceShape_[iA]) {
                    tmpS = FloorAlign(tmpS, ubBlockSize);
                }
                uint64_t tmpInnerA = innerA * tmpS;
                uint64_t tmpOuterA =
                    (tmpS >= sliceShape_[iA] ? outerA / axisLen * CeilDiv(axisLen, tmpS) :
                                            outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], tmpS));
                uint64_t tempCoreNum =
                    tmpOuterA / CeilDiv(tmpOuterA, compileInfo_.vectorCoreNum);
                tempCoreNum = tempCoreNum > tmpOuterA ? FloorAlign(tempCoreNum, tmpOuterA) : tempCoreNum;
                bool isContinue =
                    (tmpInnerA * innerR * cBlock_.aSize * cBlock_.rSize <= bBlockNum &&
                    tmpInnerA * cBlock_.aSize <= maxInnerA);
                if (isContinue) {
                    if (tmpInnerA * cBlock_.aSize <= cacheSize) {
                        maxStep = tmpS;
                    } else {
                        maxStep = tempCoreNum >= THRES_CORENUM ? tmpS : maxStep;
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
                outerA =
                    (step >= sliceShape_[iA] ? outerA / axisLen * CeilDiv(axisLen, step) :
                                            outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], step));
                break;
            }
            innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
            outerA /= axisLen;
        }
        AssembleUnit(unitA_, iA, innerA, outerA, step);
    }

    static inline void AssembleUnit(ReduceTilingUnit& unit, int32_t idx, uint64_t inner, uint64_t outer, uint64_t step)
    {
        unit.idx = idx;
        unit.inner = inner;
        unit.outer = outer;
        unit.step = step;
    }

    static void TransformShape(uint64_t* shape, int64_t* viewStride, int32_t& shapeSize)
    {
        shape[0] = 1UL;
        EliminateOne(shape, viewStride, shapeSize);
        MergeAxis(shape, viewStride, shapeSize);
    }

    static void EliminateOne(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize)
    {
        int32_t dstIdx = 1; // shape中第一个数给了1, 跳过第一个数
        for (size_t i = 0; i < reduceAxisInfo.size(); i++) {
            // 前面补了一维，所有的axes需要加1
            reduceAxisInfo[i] = reduceAxisInfo[i] + 1;
        }
        int32_t eraseNum = 0;
        for (size_t i = 0; i < shapeInfo.size(); i++) {
            auto iter = std::find(reduceAxisInfo.begin(), reduceAxisInfo.end(), i + 1);
            bool isContiguous{false};
            if (i != shapeInfo.size() - 1) {
                isContiguous = (dimStrides[i] == dimStrides[i + 1] * shapeInfo[i + 1]);
            } else {
                isContiguous = (dimStrides[i] == 1UL);
            }
            if (shapeInfo[i] != 1L || !isContiguous) {
                viewShape[dstIdx] = shapeInfo[i];
                viewStride[dstIdx] = dimStrides[i];
                if (iter != reduceAxisInfo.end()) {
                    *iter = *iter - eraseNum;
                }
                dstIdx++;
            } else {
                eraseNum++;
                if (iter != reduceAxisInfo.end()) {
                    reduceAxisInfo.erase(iter);
                }
            }
        }
        shapeSize = dstIdx;
        viewStride[0] = dimStrides[0] * shapeInfo[0];
        printf(
            "after EliminateOne, Shape is:%s, axes:%s\n",
            VectorToString(viewShape, shapeSize).c_str(),
            VectorToString(reduceAxisInfo).c_str());
    }

    static void MergeAxis(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize)
    {
        int32_t tmpSize = 0;
        for (int32_t i = 0; i < shapeSize;) {
            sliceNum_[tmpSize] = 1UL;
            sliceShape_[tmpSize] = viewShape[i];
            sliceStride_[tmpSize] = viewStride[i];
            viewStride_[tmpSize] = viewStride[i];
            auto iter0 = std::find(reduceAxisInfo.begin(), reduceAxisInfo.end(), i);
            bool isRAxis0 = iter0 != reduceAxisInfo.end();
            uint64_t s = viewShape[i];
            int32_t j = i + 1;
            for (; j < shapeSize; j++) {
                int64_t curContiguousStride = (j == shapeSize - 1 ? 1L : viewStride[j + 1] * viewShape[j + 1]);
                bool isContiguous1 = (viewStride[j] == curContiguousStride);
                auto iter1 = std::find(reduceAxisInfo.begin(), reduceAxisInfo.end(), j);
                bool isRAxis1 = iter1 != reduceAxisInfo.end();
                if (isRAxis0 != isRAxis1) {
                    break;
                }
                if (!isContiguous1) {
                    sliceNum_[tmpSize] = s; // sliceNum 更新为非连续的非尾轴大小
                }
                s *= viewShape[j];
                sliceShape_[tmpSize] =
                    isContiguous1 ? sliceShape_[tmpSize] * viewShape[j] : viewShape[j]; // sliceShape 更新为非连续的尾轴大小
                sliceStride_[tmpSize] = isContiguous1 ? curContiguousStride : sliceStride_[tmpSize];
                viewStride_[tmpSize] = viewStride[j];
                if (isRAxis1) {
                    // 连续的R轴, 需要擦除后续R轴的索引
                    reduceAxisInfo.erase(iter1);
                }
            }
            i = j;
            viewShape[tmpSize] = s;
            tmpSize++;
            if (isRAxis0) {
                *iter0 = tmpSize - 1;
            }
        }
        for (int32_t i = tmpSize; i < shapeSize; i++) {
            viewShape[i] = 0;
        }
        shapeSize = tmpSize;
        printf(
            "after merge axis, Shape:%s, axes:%s\n",
            VectorToString(viewShape, shapeSize).c_str(),
            VectorToString(reduceAxisInfo).c_str());
    }

    template <class Pattern>
    int32_t ReduceOpTiling::IsUseNddma(const uint64_t* shape)
    {
        int32_t axis = cBlock_.axis;
        uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
        uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
        if (viewStride_[Pattern::Dim - 1] != 1UL) {
            // 尾轴非连续, 做NDDMA
            return 1;
        }
        if (shape[Pattern::Dim - 1] >= ubBlockSize) {
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
                factorA = factorA * shape[iA];
            }
            if (factorA > ubBlockSize) {
                // 转置后A轴乘积大于ubblock, 不做NDDMA
                return 0;
            }
        } else {
            uint64_t factorR = uintR.step;
            for (auto iR = unitR_.idx + AXES_STEP; iR < Pattern::Dim; iR += AXES_STEP) {
                factorR = factorR * shape[iR];
            }
            if (factorR > ubBlockSize) {
                // 转置后R轴乘积大于ubblock, 不做NDDMA
                return 0;
            }
        }
        return 1;
    }

    template <class Pattern>
    static void ComputeStride(const uint64_t* shape)
    {
        uint64_t s = 1UL;
        uint64_t ds = 1UL;
        for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
            srcstride[dim] = s;
            dstStride[dim] = ds;
            s *= shape[dim];
            if (IsAxisA<Pattern>(dim)) {
                ds *= shape[dim];
            }
        }
    }

    static uint64_t Ratio()
    {
        if (opDag_.memLevel == MemLevel::LEVEL_2) {
            return CeilDiv(opDag_.maxInputBytes, static_cast<uint64_t>(ge::GetSizeByDataType(opInput_.inputDtype)));
        }
        return 1UL;
    }

    template <class Pattern>
    static bool IsAxisA(int32_t idx)
    {
        if (Pattern::FirstA) {
            return idx % CONST2 == 0;
        } else {
            return idx % CONST2 == 1;
        }
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

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    /*!
     * \brief The constructor interface of KernelBuilder class.
     */
    __aicore__ inline BaseKernelSchedule() = default;

    /*!
     * \brief Kernel layer execution function.
     * \param[in] cfg, Tiling information in kernel.
     * \param[in] argTuple, Input and output GM address.
     */
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        // Configure block scheduling parameters
        typename BlockOp::ScheduleClz::ParamStruct configBlock = cfg.blockParam;
        configBlock.totalElemCnt = CalCurCoreEleCnt(cfg.kernelParam);
        // Pack and prepare arguments for kernel execution
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);
        auto params = PrepareParams<Params>(cfg.kernelParam, argTuple);
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

template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class DefaultKernelSchedule : public BaseKernelSchedule<BlockOp, Policy, ScheduleCfg> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
