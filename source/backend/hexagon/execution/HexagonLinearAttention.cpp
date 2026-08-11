#include "HexagonLinearAttention.hpp"

#include <cstring>
#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "core/KVMeta.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "htp_command.h"

namespace MNN {

struct HexagonLinearAttention::State {
    std::shared_ptr<Tensor> conv;
    std::shared_ptr<Tensor> recurrent;
    size_t convBytes = 0;
    size_t recurrentBytes = 0;
    Backend* backend = nullptr;

    ~State() {
        if (backend == nullptr) {
            return;
        }
        if (conv != nullptr) {
            backend->onReleaseBuffer(conv.get(), Backend::STATIC);
        }
        if (recurrent != nullptr) {
            backend->onReleaseBuffer(recurrent.get(), Backend::STATIC);
        }
    }
};

static bool isC4(const Tensor* tensor) {
    return tensor != nullptr && TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
}

static void linearDims(const Tensor* qkv, int& batch, int& channels, int& sequence) {
    if (isC4(qkv)) {
        batch = 1;
        sequence = qkv->length(0);
        channels = qkv->length(1);
    } else {
        batch = qkv->length(0);
        channels = qkv->length(1);
        sequence = qkv->length(2);
    }
}

HexagonLinearAttention::HexagonLinearAttention(Backend* backend, const Op* op, std::shared_ptr<State> state)
    : HexagonExecution(backend), mState(std::move(state)) {
    const auto param = op->main_as_LinearAttentionParam();
    if (param != nullptr) {
        mNumKHeads = param->num_k_heads();
        mNumVHeads = param->num_v_heads();
        mHeadKDim = param->head_k_dim();
        mHeadVDim = param->head_v_dim();
        mUseQKL2Norm = param->use_qk_l2norm();
    }
    mPack = static_cast<const HexagonRuntime*>(backend->getRuntime())->info().vectorSize;
    if (mPack <= 0) {
        mPack = 4;
    }
    mMeta = reinterpret_cast<KVMeta*>(backend->getMetaPtr());
#if defined(MNN_HEXAGON_OFFLINE_RPC) || defined(MNN_GPU_TIME_PROFILE)
    setDebugName(op->name() == nullptr ? "LinearAttention" : op->name()->c_str());
#endif
}

HexagonLinearAttention::~HexagonLinearAttention() = default;

ErrorCode HexagonLinearAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Tuning may execute the prepared command graph before the first real
    // request. Reset only at the same request boundary used by the CPU
    // implementation; decode executions must preserve both states.
    if (mSequence > 1 && (mMeta == nullptr || mMeta->previous == mMeta->remove) && mState != nullptr) {
        if (mState->conv != nullptr) {
            ::memset(HexagonBackend::getPtr(mState->conv.get()), 0, mState->convBytes);
        }
        if (mState->recurrent != nullptr) {
            ::memset(HexagonBackend::getPtr(mState->recurrent.get()), 0, mState->recurrentBytes);
        }
    }
    return HexagonExecution::onExecute(inputs, outputs);
}

HexagonLinearAttention* HexagonLinearAttention::create(Backend* backend, const Op* op) {
    if (op == nullptr || op->type() != OpType_LinearAttention) {
        return nullptr;
    }
    const auto* param = op->main_as_LinearAttentionParam();
    if (param == nullptr || param->attn_type() == nullptr || param->attn_type()->str() != "gated_delta_rule") {
        return nullptr;
    }
    return new HexagonLinearAttention(backend, op, std::shared_ptr<State>(new State));
}

bool HexagonLinearAttention::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    auto execution = new HexagonLinearAttention(backend, op, mState);
    execution->mBatch = mBatch;
    execution->mConvDim = mConvDim;
    execution->mSequence = mSequence;
    execution->mNumKHeads = mNumKHeads;
    execution->mNumVHeads = mNumVHeads;
    execution->mHeadKDim = mHeadKDim;
    execution->mHeadVDim = mHeadVDim;
    execution->mConvKernel = mConvKernel;
    execution->mQKVC4 = mQKVC4;
    execution->mGateC4 = mGateC4;
    execution->mBetaC4 = mBetaC4;
    execution->mOutputC4 = mOutputC4;
    execution->mWeightC4 = mWeightC4;
    execution->mUseQKL2Norm = mUseQKL2Norm;
    execution->mPack = mPack;
    execution->mMeta = mMeta;
    *dst = execution;
    return true;
}

ErrorCode HexagonLinearAttention::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                             std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 4 || outputs.size() != 1 || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || inputs[3] == nullptr || outputs[0] == nullptr || inputs[3]->dimensions() < 3) {
        mValid = false;
        return INPUT_DATA_ERROR;
    }
    if (HexagonBackend::getBytes(outputs[0]) != 2 || HexagonBackend::getBytes(inputs[0]) != 2) {
        mValid = false;
        return NOT_SUPPORT;
    }

    linearDims(inputs[0], mBatch, mConvDim, mSequence);
    mConvKernel = inputs[3]->length(2);
    if (mBatch <= 0 || mConvDim <= 0 || mSequence <= 0 || mConvKernel <= 0 || mNumKHeads <= 0 || mNumVHeads <= 0 ||
        mNumVHeads % mNumKHeads != 0 || mHeadKDim <= 0 || mHeadVDim <= 0 ||
        mConvDim != 2 * mNumKHeads * mHeadKDim + mNumVHeads * mHeadVDim) {
        mValid = false;
        return NOT_SUPPORT;
    }
    mQKVC4 = isC4(inputs[0]);
    mGateC4 = isC4(inputs[1]);
    mBetaC4 = isC4(inputs[2]);
    mOutputC4 = isC4(outputs[0]);
    mWeightC4 = isC4(inputs[3]);

    constexpr size_t kFp16Bytes = 2;
    const size_t convBytes = (size_t)mBatch * mConvDim * (mConvKernel - 1) * kFp16Bytes;
    const size_t recurrentBytes = (size_t)mBatch * mNumVHeads * mHeadKDim * mHeadVDim * kFp16Bytes;
    if (mState->conv == nullptr || mState->recurrent == nullptr || mState->convBytes != convBytes ||
        mState->recurrentBytes != recurrentBytes) {
        if (mState->conv != nullptr) {
            backend()->onReleaseBuffer(mState->conv.get(), Backend::STATIC);
            mState->conv.reset();
        }
        if (mState->recurrent != nullptr) {
            backend()->onReleaseBuffer(mState->recurrent.get(), Backend::STATIC);
            mState->recurrent.reset();
        }
        mState->backend = backend();
        mState->conv.reset(Tensor::createDevice<int8_t>({(int)convBytes}));
        mState->recurrent.reset(Tensor::createDevice<int8_t>({(int)recurrentBytes}));
        if (!backend()->onAcquireBuffer(mState->conv.get(), Backend::STATIC) ||
            !backend()->onAcquireBuffer(mState->recurrent.get(), Backend::STATIC)) {
            mValid = false;
            return OUT_OF_MEMORY;
        }
        mState->convBytes = convBytes;
        mState->recurrentBytes = recurrentBytes;
        ::memset(HexagonBackend::getPtr(mState->conv.get()), 0, convBytes);
        ::memset(HexagonBackend::getPtr(mState->recurrent.get()), 0, recurrentBytes);
    } else if (mSequence > 1 && (mMeta == nullptr || mMeta->previous == mMeta->remove)) {
        ::memset(HexagonBackend::getPtr(mState->conv.get()), 0, convBytes);
        ::memset(HexagonBackend::getPtr(mState->recurrent.get()), 0, recurrentBytes);
    }

    const size_t scratchBytes = (size_t)mBatch * mSequence * mConvDim * kFp16Bytes;
    mConvScratch.reset(Tensor::createDevice<int8_t>({(int)scratchBytes}));
    if (!backend()->onAcquireBuffer(mConvScratch.get(), Backend::DYNAMIC)) {
        mValid = false;
        return OUT_OF_MEMORY;
    }

    int params[] = {mBatch,          mConvDim,          mSequence,         mNumKHeads,           mNumVHeads,
                    mHeadKDim,       mHeadVDim,         mConvKernel,       mQKVC4 ? 1 : 0,       mGateC4 ? 1 : 0,
                    mBetaC4 ? 1 : 0, mOutputC4 ? 1 : 0, mWeightC4 ? 1 : 0, mUseQKL2Norm ? 1 : 0, mPack};
    std::vector<std::pair<int, int>> inputFds = {
        HexagonBackend::getDevicePtr(inputs[0]),          HexagonBackend::getDevicePtr(inputs[1]),
        HexagonBackend::getDevicePtr(inputs[2]),          HexagonBackend::getDevicePtr(inputs[3]),
        HexagonBackend::getDevicePtr(mState->conv.get()), HexagonBackend::getDevicePtr(mState->recurrent.get())};
    std::vector<std::pair<int, int>> outputFds = {
        HexagonBackend::getDevicePtr(outputs[0]), HexagonBackend::getDevicePtr(mState->conv.get()),
        HexagonBackend::getDevicePtr(mState->recurrent.get()), HexagonBackend::getDevicePtr(mConvScratch.get())};
    std::vector<Tensor*> commandInputs = {inputs[0], inputs[1],          inputs[2],
                                          inputs[3], mState->conv.get(), mState->recurrent.get()};
    std::vector<Tensor*> commandOutputs = {outputs[0], mState->conv.get(), mState->recurrent.get(), mConvScratch.get()};
    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_LINEAR_ATTENTION_GATED_DELTA, params,
                     sizeof(params), inputFds, outputFds, commandInputs, commandOutputs);
    backend()->onReleaseBuffer(mConvScratch.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

} // namespace MNN
