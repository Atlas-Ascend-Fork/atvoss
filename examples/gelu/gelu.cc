#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"
#include "utils/layout/shape.h"

static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;

template <typename T1, typename T2, typename T3>
struct GeluConfig {
    using DtypeV1 = T1;
    using DtypeV2 = T2;
    using DtypeV3 = T3;
    using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;

    struct GeluCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::in>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<DtypeV2>, Atvoss::ParamUsage::in>();
            auto out = Atvoss::PlaceHolder<3, Tensor<DtypeV3>, Atvoss::ParamUsage::out>();
            auto temp = Atvoss::PlaceHolderTmpLike<1>(in1);

            return (temp = in1 * in1,
                    out = ReduceSum<Atvoss::Pattern::AR>(temp),
                    out = Broadcast<Atvoss::Pattern::AB>(out),
                    temp = Divs<WIDTH>(out),
                    out = Sqrt(temp),
                    temp = in1 / out,
                    out = in2 * temp);
        }
    };

    static constexpr Atvoss::EleWise::BlockPolicy<TileShape> blockPolicy {
        190 * 1024,
        TileShape{}
    };
    static constexpr Atvoss::EleWise::KernelPolicy kernelPolicy {
        48,
        Atvoss::EleWise::KernelPolicySegment::UniformSegment
    };

    using BlockOp = Atvoss::EleWise::BlockBuilder<
        GeluCompute,
        blockPolicy,
        Atvoss::EleWise::BlockConfig>;

    using KernelOp = Atvoss::EleWise::KernelBuilder<
        BlockOp,
        kernelPolicy>;

    // using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

template<class KernelOp, typename OpParam>
__global__ __aicore__ void gate_gelu_odd_kernel(OpParam cfg, GM_ADDR x)
{
    AscendC::TPipe pipeIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x);
}

// ===================== Kernel wrapper =====================

// extern "C" __global__ __aicore__
// void gate_gelu_odd_kernel(int32_t pre_channel_dims,
//                           int32_t channel,
//                           GM_ADDR x,
//                           GM_ADDR z)
// {
//     GATE_GELU_Odd op;
//     op.Init(pre_channel_dims, channel, x, z);
//     op.Process();
// }

#ifndef __CCE_KT_TEST__
// host 侧 launch 接口
void gate_gelu_odd_launch(uint32_t blockDim,
                          void* stream,
                          uint32_t pre_channel_dims,
                          uint32_t channel,
                          unsigned char* x,
                          unsigned char* out)
{
    KernelCustom<GeluConfig::KernelOp><<<blockDim, nullptr, stream>>>(
        (int32_t)pre_channel_dims,
        (int32_t)channel,
        (GM_ADDR)x,
        (GM_ADDR)out);
}
#endif


// ===================== onnxinfer kernel 注册 =====================

template <typename T>
class AddGateGeluOdd final : public AtlasKernel {
public:
    AddGateGeluOdd(const OpKernelInfo& op_kernel_info)
        : AtlasKernel(op_kernel_info) {}

    ~AddGateGeluOdd() {}

    Status ComputeInternal(OpKernelContext* ctx) const override
    {
        gate_gelu_odd_launch(
            40,
            stream(),
            (uint32_t)(b * s),
            (uint32_t)c,
            (unsigned char*)(const_cast<void*>(inputx->DataRaw())),
            (unsigned char*)(const_cast<void*>(out)));

        return Status::OK();
    }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AddGateGeluOdd,
    kOnnxDomain,
    1,
    MLFLOAT16,
    AtlasExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFLOAT16>()),
    AddGateGeluOdd<MLFLOAT16>);
};

