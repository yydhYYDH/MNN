#ifndef HexagonExecution_hpp
#define HexagonExecution_hpp

#include <vector>
#ifdef MNN_HEXAGON_OFFLINE_RPC
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
#ifdef MNN_HEXAGON_OFFLINE_RPC
    void setOfflineDebugName(const char* name) {
        mOfflineDebugName = name != nullptr ? name : "";
    }
#endif

protected:
    virtual ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 std::vector<HexagonCommand>& dst) = 0;

    std::vector<HexagonCommand> mCmd;
#ifdef MNN_HEXAGON_OFFLINE_RPC
    std::string mOfflineDebugName;
#endif
};

} // namespace MNN

#endif
