#include "HexagonTopKV2.hpp"

#include <numeric>

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "htp_command.h"

namespace MNN {

namespace {
constexpr int32_t kMaxHexagonTopK = 64;
}

HexagonTopKV2::HexagonTopKV2(Backend* backend) : HexagonExecution(backend) {
}

HexagonTopKV2* HexagonTopKV2::create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    if (op == nullptr || op->type() != OpType_TopKV2 || inputs.size() != 2 || outputs.size() != 2) {
        return nullptr;
    }
    auto param = op->main_as_TopKV2();
    if (param != nullptr && !param->largest()) {
        return nullptr;
    }
    if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr || outputs[1] == nullptr) {
        return nullptr;
    }
    if (inputs[0]->getType().code != halide_type_float || outputs[0]->getType().code != halide_type_float ||
        outputs[1]->getType().code != halide_type_int || outputs[1]->getType().bits != 32) {
        return nullptr;
    }
    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    auto valueDes = TensorUtils::getDescribe(outputs[0]);
    auto indexDes = TensorUtils::getDescribe(outputs[1]);
    const auto isSupportedFormat = [](MNN_DATA_FORMAT format) {
        return format == MNN_DATA_FORMAT_NCHW || format == MNN_DATA_FORMAT_NHWC ||
               format == MNN_DATA_FORMAT_NC4HW4 || format == MNN_DATA_FORMAT_NHWC4 ||
               format == MNN_DATA_FORMAT_UNKNOWN;
    };
    if (!isSupportedFormat(inputDes->dimensionFormat) || !isSupportedFormat(valueDes->dimensionFormat) ||
        !isSupportedFormat(indexDes->dimensionFormat)) {
        return nullptr;
    }
    if (HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    return new HexagonTopKV2(backend);
}

ErrorCode HexagonTopKV2::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 2 || outputs.size() != 2) {
        return NOT_SUPPORT;
    }
    const auto input = inputs[0];
    const auto values = outputs[0];
    const auto indices = outputs[1];
    const int dimensions = input->dimensions();
    if (dimensions <= 0) {
        return NOT_SUPPORT;
    }
    const int rowSize = input->length(dimensions - 1);
    if (rowSize <= 0 || input->elementSize() % rowSize != 0) {
        return NOT_SUPPORT;
    }
    const int rows = input->elementSize() / rowSize;
    if (rows <= 0 || values->elementSize() <= 0 || values->elementSize() % rows != 0 ||
        indices->elementSize() != values->elementSize()) {
        return NOT_SUPPORT;
    }
    const int k = values->elementSize() / rows;
    if (k <= 0 || k > kMaxHexagonTopK || k > rowSize) {
        return NOT_SUPPORT;
    }
    const int inputBytes = HexagonBackend::getBytes(input);
    const int valueBytes = HexagonBackend::getBytes(values);
    const int indexBytes = HexagonBackend::getBytes(indices);
    if (inputBytes != 2 || (valueBytes != 2 && (valueBytes != 4 || k == 1)) || indexBytes != 4) {
        return NOT_SUPPORT;
    }

    auto inputDev = HexagonBackend::getDevicePtr(input);
    auto valueDev = HexagonBackend::getDevicePtr(values);
    auto indexDev = HexagonBackend::getDevicePtr(indices);

    int32_t params[4] = {rowSize, rows, k, valueBytes};
    std::vector<std::pair<int, int>> inputFds = {inputDev};
    std::vector<std::pair<int, int>> outputFds = {valueDev, indexDev};

    dst.emplace_back();
    const int opType = k == 1 ? DSP_OP_TOPKV2_K1_FP16 : DSP_OP_TOPKV2_FP16;
    const size_t paramSize = k == 1 ? sizeof(int32_t) * 2 : sizeof(params);
    dst.back().build(static_cast<HexagonBackend*>(backend()), opType, params, paramSize,
                     inputFds, outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonTopKV2::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonTopKV2(bn);
    return true;
}

} // namespace MNN
