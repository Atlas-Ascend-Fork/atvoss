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

#ifndef _LOOP_BUFFER_POOL_H_
#define _LOOP_BUFFER_POOL_H_

namespace Atvoss {

struct BufPoolCustom {
    uint64_t startAddr;
    uint64_t maxAddr;
    uint64_t maxLen;
};

template<AscendC::TPosition src, AscendC::TPosition dst, int tileNum, int tileSize>
class LoopBufferEx {
private:
    static_assert(tileSize % 1024 == 0, "[ERROR]: [Atvoss][LoopBuffer] Tile must be a multiple of 1024\n");
    static_assert(tileNum >= 0 && tileNum <= 8, "[ERROR]: [Atvoss][LoopBuffer] TileNum must be [0, 8]\n");

    static __aicore__ constexpr pipe_t ToPipe(AscendC::TPosition p){
        if( p == AscendC::TPosition::VECIN) {
            return PIPE_MTE2;
        }
        else if( p== AscendC::TPosition::VECOUT){
            return PIPE_MTE3;
        }
        else if(p==AscendC::TPosition::VECCALC) {
            return PIPE_V;
        }
    }
    static __aicore__ constexpr AscendC::HardEvent ToEvent(){
        if( src== AscendC::TPosition::VECIN && dst==AscendC::TPosition::VECCALC){
            return AscendC::HardEvent::MTE2_V;
        }
        else if( src== AscendC::TPosition::VECIN && dst==AscendC::TPosition::VECOUT){
            return AscendC::HardEvent::MTE2_MTE3;
        }
        else if( src== AscendC::TPosition::VECCALC && dst==AscendC::TPosition::VECOUT ) {
            return AscendC::HardEvent::V_MTE3;
        }
		return AscendC::HardEvent::V_V;
    }
    static __aicore__ constexpr AscendC::HardEvent ToRevEvent() {
        if( src== AscendC::TPosition::VECIN && dst==AscendC::TPosition::VECCALC){
            return AscendC::HardEvent::V_MTE2;
        }
        else if( src== AscendC::TPosition::VECIN && dst==AscendC::TPosition::VECOUT){
            return AscendC::HardEvent::MTE3_MTE2;
        }
        else if( src== AscendC::TPosition::VECCALC && dst==AscendC::TPosition::VECOUT ) {
            return AscendC::HardEvent::MTE3_V;
        }
		return AscendC::HardEvent::V_V;
    }
public:
    __aicore__ inline LoopBufferEx(){
    }
    __aicore__ inline ~LoopBufferEx(){}

    __aicore__ inline void DeInit() {
        for (uint8_t i = 0; i < tileNum; i++) {
            auto evtID = bufState[i].evtID;
            if (evtID >= 0) {
                wait_flag((pipe_t)ToPipe(dst), (pipe_t)ToPipe(src), evtID);
            }
        }
    }
    template<size_t baseAddr>
    __aicore__ inline void Init() {
        if constexpr (tileNum == 0) {
            return;
        }
        ASCENDC_ASSERT((baseAddr % 32 == 0),  { KERNEL_LOG(KERNEL_ERROR, "[ERROR]: [Atvoss][LoopBuffer] InitBuffer bufferSize must be free state."); });
        bufPool_.startAddr = baseAddr;
        bufPool_.maxAddr = baseAddr + POOL_LEN;
        bufPool_.maxLen = POOL_LEN;
        GetTPipePtr()->InitBuffer(tbuf_, POOL_LEN);
        this->header  = 0;
#pragma unroll
        for(int i=0; i<tileNum; i++) {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            bufState[i].state = 0;
#endif
            bufState[i].evtID = -1;
        }
    }

    template <typename T, class... Args>
    __aicore__ inline void AllocTensor(AscendC::LocalTensor<T>& tensor, Args... args){
        if constexpr (tileNum == 0) {
            return;
        }
        AscendC::TBuffAddr addr;
        addr.bufferAddr = bufPool_.startAddr + header * tileSize;
        addr.logicPos =  (uint8_t)AscendC::TPosition::VECCALC;
        addr.bufferHandle = (AscendC::TBufHandle)header; 
        addr.dataLen = tileSize;
		if constexpr (ToRevEvent() != AscendC::HardEvent::V_V) {
	        if( bufState[header].evtID >= 0 ){ 
	            auto evtID = bufState[header].evtID;
	            wait_flag((pipe_t)ToPipe(dst), (pipe_t)ToPipe(src), evtID);
	            GetTPipePtr()->ReleaseEventID<ToRevEvent()>(evtID);
	            bufState[header].evtID = -1;
	        }
		}

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT(( bufState[header].state == 0),  { KERNEL_LOG(KERNEL_ERROR, "[ERROR]: [Atvoss][LoopBuffer] buffer must be free state."); });
        bufState[header].state = 1;
        addr.absAddr = nullptr;
#endif

        if( ++header >= tileNum ) {
            header = 0;
        }
        tensor.SetAddr(addr);

        if constexpr( sizeof...(args) > 0) {
            AllocTensor(args...);
        }
    }
    template <typename T, class... Args>
    __aicore__ inline void FreeTensor(T& tensor, Args... args){
        if constexpr (tileNum == 0) {
            return;
        }
        auto bufPos = static_cast<uint8_t>(reinterpret_cast<uintptr_t>(tensor.GetBufferHandle()));
		if constexpr (ToRevEvent() != AscendC::HardEvent::V_V) {
	        auto evtID = GetTPipePtr()->AllocEventID< ToRevEvent()>();
	        bufState[bufPos].evtID = evtID;
	        set_flag((pipe_t)ToPipe(dst), (pipe_t)ToPipe(src), evtID);
		}

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        ASCENDC_ASSERT(( bufPos >= 0 && bufPos < tileNum),  { KERNEL_LOG(KERNEL_ERROR, ""); });
        ASCENDC_ASSERT(( bufState[bufPos].state == 1),  { KERNEL_LOG(KERNEL_ERROR, "[ERROR]: [Atvoss][LoopBuffer] buffer must be alloc state."); });
        bufState[bufPos].state = 0;
#endif
        if constexpr( sizeof...(args) > 0) {
            FreeTensor(args...);
        }
    }

    static __aicore__ inline void Sync(){
		if constexpr (ToEvent() != AscendC::HardEvent::V_V) {
            auto evtID = GetTPipePtr()->FetchEventID< ToEvent()>();
            set_flag((pipe_t)ToPipe(src), (pipe_t)ToPipe(dst), evtID);
            wait_flag((pipe_t)ToPipe(src), (pipe_t)ToPipe(dst), evtID);
		}
    }
private:
    BufPoolCustom bufPool_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf_;
    static constexpr uint64_t POOL_LEN = tileSize * tileNum;
    // uint64_t baseAddr;
    uint8_t  header;
    struct BufState {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        uint8_t state;
#endif
        int8_t evtID;
    } bufState[tileNum];
};

template<AscendC::TPosition pos, int tileNum, int tileSize>
class LoopBuffer{};

template<int tileNum, int tileSize>
class LoopBuffer<AscendC::TPosition::VECIN, tileNum, tileSize>
    : public LoopBufferEx<AscendC::TPosition::VECIN, AscendC::TPosition::VECCALC, tileNum, tileSize> {
};
template<int tileNum, int tileSize>
class LoopBuffer<AscendC::TPosition::VECOUT, tileNum, tileSize>
    : public LoopBufferEx<AscendC::TPosition::VECCALC, AscendC::TPosition::VECOUT, tileNum, tileSize> {
};
template<int tileNum, int tileSize>
class LoopBuffer<AscendC::TPosition::VECCALC, tileNum, tileSize>
    : public LoopBufferEx<AscendC::TPosition::VECCALC, AscendC::TPosition::VECCALC, tileNum, tileSize> {
};
} //namespace Atvoss

#endif //_LOOP_BUFFER_POOL_H_