#ifndef HexagonExecution_hpp
#define HexagonExecution_hpp

#include <vector>
#if defined(MNN_HEXAGON_OFFLINE_RPC) || defined(MNN_GPU_TIME_PROFILE)
#include <string>
#endif

#include "core/Execution.hpp"
#include "backend/hexagon/backend/HexagonCommand.hpp"

namespace MNN {

class HexagonExecution : public Execution {
public:
    explicit HexagonExecution(Backend* backend);
    virtual ~HexagonExecution() = default;

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
#if defined(MNN_HEXAGON_OFFLINE_RPC) || defined(MNN_GPU_TIME_PROFILE)
    void setDebugName(const char* name) {
        mDebugName = name != nullptr ? name : "";
    }
#endif

protected:
    virtual ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 std::vector<HexagonCommand>& dst) = 0;

    std::vector<HexagonCommand> mCmd;
#if defined(MNN_HEXAGON_OFFLINE_RPC) || defined(MNN_GPU_TIME_PROFILE)
    std::string mDebugName;
#endif
};

} // namespace MNN

#endif
