/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file loopbuf.h
 * \brief loopbuf api of AscendC
eg1：
    LoopBuffer<TPosition::VECIN, 2, 1024>  queIn;

    LocalTensr<int> a;
    LocalTensr<float> b;
    LocalTensr<float> c;

    queIn.AllocTensor(a, b);
    queIn.AllocTensor(c);

    ...
    queIn.FreeTensor(a, b);
    queIn.FreeTensor(c);
 */

#ifndef ATVOSS_LOOP_BUFFER_POOL_H
#define ATVOSS_LOOP_BUFFER_POOL_H

namespace Atvoss {

struct BufPoolCustom {
    uint64_t startAddr;
    uint64_t maxAddr;
    uint64_t maxLen;
};

template <AscendC::TPosition src, AscendC::TPosition dst, int tileNum, int tileSize, bool useTPipe>
class LoopBufferEx {
private:
    static_assert(tileSize % 1024 == 0, "[ERROR]: [Atvoss][LoopBuffer] Tile must be a multiple of 1024\n");
    static_assert(tileNum >= 0 && tileNum <= 8, "[ERROR]: [Atvoss][LoopBuffer] TileNum must be [0, 8]\n");

    static __aicore__ constexpr pipe_t ToPipe(AscendC::TPosition p)
    {
        if (p == AscendC::TPosition::VECIN) {
            return PIPE_MTE2;
        } else if (p == AscendC::TPosition::VECOUT) {
            return PIPE_MTE3;
        } else if (p == AscendC::TPosition::VECCALC) {
            return PIPE_V;
        }
    }
    static __aicore__ constexpr AscendC::HardEvent ToEvent()
    {
        if (src == AscendC::TPosition::VECIN && dst == AscendC::TPosition::VECCALC) {
            return AscendC::HardEvent::MTE2_V;
        } else if (src == AscendC::TPosition::VECIN && dst == AscendC::TPosition::VECOUT) {
            return AscendC::HardEvent::MTE2_MTE3;
        } else if (src == AscendC::TPosition::VECCALC && dst == AscendC::TPosition::VECOUT) {
            return AscendC::HardEvent::V_MTE3;
        }
        return AscendC::HardEvent::V_V;
    }
    static __aicore__ constexpr AscendC::HardEvent ToRevEvent()
    {
        if (src == AscendC::TPosition::VECIN && dst == AscendC::TPosition::VECCALC) {
            return AscendC::HardEvent::V_MTE2;
        } else if (src == AscendC::TPosition::VECIN && dst == AscendC::TPosition::VECOUT) {
            return AscendC::HardEvent::MTE3_MTE2;
        } else if (src == AscendC::TPosition::VECCALC && dst == AscendC::TPosition::VECOUT) {
            return AscendC::HardEvent::MTE3_V;
        }
        return AscendC::HardEvent::V_V;
    }

public:
    __aicore__ inline LoopBufferEx()
    {}
    __aicore__ inline ~LoopBufferEx()
    {}

    __aicore__ inline void DeInit()
    {
        for (uint8_t i = 0; i < tileNum; i++) {
            auto evtID = bufState_[i].evtID_;
            if (evtID >= 0) {
                wait_flag((pipe_t)ToPipe(dst), (pipe_t)ToPipe(src), evtID);
            }
        }
    }
    template <size_t baseAddr>
    __aicore__ inline void Init()
    {
        if constexpr (tileNum == 0) {
            return;
        }
        ASCENDC_ASSERT((baseAddr % 32 == 0), {
            KERNEL_LOG(KERNEL_ERROR, "[ERROR]: [Atvoss][LoopBuffer] InitBuffer bufferSize must be free state.");
        });
        bufPool_.startAddr = baseAddr;
        bufPool_.maxAddr = baseAddr + POOL_LEN;
        bufPool_.maxLen = POOL_LEN;
        if constexpr (useTPipe) {
            GetTPipePtr()->InitBuffer(tbuf_, POOL_LEN);
        }
        this->header_ = 0;
        for (int i = 0; i < tileNum; i++) {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            bufState_[i].state_ = 0;
#endif
            bufState_[i].evtID_ = -1;
        }
    }

    template <typename T, class... Args>
    __aicore__ inline void AllocTensor(AscendC::LocalTensor<T>& tensor, Args... args)
    {
        if constexpr (tileNum == 0) {
            return;
        }
        AscendC::TBuffAddr addr;
        addr.bufferAddr = bufPool_.startAddr + header_ * tileSize;
        addr.logicPos = (uint8_t)AscendC::TPosition::VECCALC;
        addr.bufferHandle = (AscendC::TBufHandle)header_;
        addr.dataLen = tileSize;
        if constexpr (ToRevEvent() != AscendC::HardEvent::V_V) {
            if (bufState_[header_].evtID_ >= 0) {
                auto evtID = bufState_[header_].evtID_;
                wait_flag((pipe_t)ToPipe(dst), (pipe_t)ToPipe(src), evtID);
                if constexpr (useTPipe) {
                    GetTPipePtr()->ReleaseEventID<ToRevEvent()>(evtID);
                }
                bufState_[header_].evtID_ = -1;
            }
        }

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((bufState_[header_].state_ == 0), {
            KERNEL_LOG(KERNEL_ERROR, "[ERROR]: [Atvoss][LoopBuffer] buffer must be free state.");
        });
        bufState_[header_].state_ = 1;
        AscendC::LocalTensor<uint32_t> tmpTensor = tbuf_.Get<uint32_t>();
        addr.absAddr = reinterpret_cast<uint8_t*>(tmpTensor.GetPhyAddr() + header_ * tileSize);
#endif

        if (++header_ >= tileNum) {
            header_ = 0;
        }
        tensor.SetAddr(addr);
        if constexpr (sizeof...(args) > 0) {
            AllocTensor(args...);
        }
    }
    template <typename T, class... Args>
    __aicore__ inline void FreeTensor(T& tensor, Args... args)
    {
        auto bufPos = static_cast<uint8_t>(reinterpret_cast<uintptr_t>(tensor.GetBufferHandle()));
        if constexpr (tileNum == 0) {
            return;
        }
        if constexpr (ToRevEvent() != AscendC::HardEvent::V_V) {
            auto evtID = 0;
            if constexpr (useTPipe) {
                evtID = GetTPipePtr()->AllocEventID<ToRevEvent()>();
            } else {
                eventID_ = eventID_ ^ 1;
                evtID = eventID_;
            }
            bufState_[bufPos].evtID_ = evtID;
            set_flag((pipe_t)ToPipe(dst), (pipe_t)ToPipe(src), evtID);
        }

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT((bufPos >= 0 && bufPos < tileNum), { KERNEL_LOG(KERNEL_ERROR, ""); });
        ASCENDC_ASSERT((bufState_[bufPos].state_ == 1), {
            KERNEL_LOG(KERNEL_ERROR, "[ERROR]: [Atvoss][LoopBuffer] buffer must be alloc state.");
        });
        bufState_[bufPos].state_ = 0;
#endif
        if constexpr (sizeof...(args) > 0) {
            FreeTensor(args...);
        }
    }

    static __aicore__ inline void Sync()
    {
        if constexpr (ToEvent() != AscendC::HardEvent::V_V) {
            auto evtID = 7;
            if constexpr (useTPipe) {
                evtID = GetTPipePtr()->FetchEventID<ToEvent()>();
            }
            set_flag((pipe_t)ToPipe(src), (pipe_t)ToPipe(dst), evtID);
            wait_flag((pipe_t)ToPipe(src), (pipe_t)ToPipe(dst), evtID);
        }
    }

private:
    int8_t eventID_ = 6;
    BufPoolCustom bufPool_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf_;
    static constexpr uint64_t POOL_LEN = tileSize * tileNum;
    uint8_t header_;
    struct BufState {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        uint8_t state_;
#endif
        int8_t evtID_;
    } bufState_[tileNum];
};

template <AscendC::TPosition pos, int tileNum, int tileSize, bool useTPipe = false>
class LoopBuffer {};

template <int tileNum, int tileSize, bool useTPipe>
class LoopBuffer<AscendC::TPosition::VECIN, tileNum, tileSize, useTPipe>
    : public LoopBufferEx<AscendC::TPosition::VECIN, AscendC::TPosition::VECCALC, tileNum, tileSize, useTPipe> {};
template <int tileNum, int tileSize, bool useTPipe>
class LoopBuffer<AscendC::TPosition::VECOUT, tileNum, tileSize, useTPipe>
    : public LoopBufferEx<AscendC::TPosition::VECCALC, AscendC::TPosition::VECOUT, tileNum, tileSize, useTPipe> {};
template <int tileNum, int tileSize, bool useTPipe>
class LoopBuffer<AscendC::TPosition::VECCALC, tileNum, tileSize, useTPipe>
    : public LoopBufferEx<AscendC::TPosition::VECCALC, AscendC::TPosition::VECCALC, tileNum, tileSize, useTPipe> {};
} // namespace Atvoss

#endif // ATVOSS_LOOP_BUFFER_POOL_H