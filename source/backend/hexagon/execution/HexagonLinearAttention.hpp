#ifndef HexagonLinearAttention_hpp
#define HexagonLinearAttention_hpp

#include "HexagonExecution.hpp"

namespace MNN {

struct KVMeta;

class HexagonLinearAttention : public HexagonExecution {
public:
    static HexagonLinearAttention* create(Backend* backend, const Op* op);
    bool onClone(Backend* backend, const Op* op, Execution** dst) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ~HexagonLinearAttention() override;

private:
    struct State;
    HexagonLinearAttention(Backend* backend, const Op* op, std::shared_ptr<State> state);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    std::shared_ptr<State> mState;
    int mBatch = 0;
    int mConvDim = 0;
    int mSequence = 0;
    int mNumKHeads = 0;
    int mNumVHeads = 0;
    int mHeadKDim = 0;
    int mHeadVDim = 0;
    int mConvKernel = 0;
    bool mQKVC4 = false;
    bool mGateC4 = false;
    bool mBetaC4 = false;
    bool mOutputC4 = false;
    bool mWeightC4 = false;
    bool mUseQKL2Norm = false;
    int mPack = 4;
    KVMeta* mMeta = nullptr;
    std::shared_ptr<Tensor> mConvScratch;
};

} // namespace MNN

#endif
