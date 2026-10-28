#ifndef HexagonAttention_hpp
#define HexagonAttention_hpp

#include "core/OpCommonUtils.hpp"
#include "MNN_generated.h"
#include "HexagonExecution.hpp"
#include "HexagonKVCacheManager.hpp"

namespace MNN {

class HexagonAttention : public HexagonExecution {
public:
    HexagonAttention(Backend *backend, const Op *op);
    virtual ~HexagonAttention();
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    static Execution* create(Backend* backend, const Op* op) {
        // The DSP path reads K/V out of a paged KV cache, so every kernel here assumes the
        // NC4HW4 cache layout. A cacheless attention (a block-diffusion draft attends a whole
        // context in one shot) arrives as plain NCHW and has no matching kernel; refuse it at
        // creation so the op falls back instead of failing session resize.
        auto param = op == nullptr ? nullptr : op->main_as_AttentionParam();
        if (param != nullptr && !param->kv_cache()) {
            return nullptr;
        }
        return new HexagonAttention(backend, op);
    }
private:
    ErrorCode onBuildCmd(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                         std::vector<HexagonCommand>& dst) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode ensurePageTableCapacity(int pageCount);
    void updatePageTable();

    float attentionScale(int headDim) const;
    std::shared_ptr<Tensor> mWorkspace;
    std::shared_ptr<Tensor> mPageTable;
    bool mUseGeneratedCausalMask = false;
    bool mVisionState = false;
    std::shared_ptr<HexagonKVCacheManager> mKVCacheManager;
    KVMeta* mMeta = nullptr;
    int mMaxKVLen = 0;
    int mPageTableCapacity = 0;
    uint64_t mSyncedPageGeneration = (uint64_t)-1;
};

} // namespace MNN

#endif /* HexagonAttention_hpp */
