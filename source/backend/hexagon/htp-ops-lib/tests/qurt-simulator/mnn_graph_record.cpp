#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include "offline_rpc_request_parser.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct InputFile {
    std::string name;
    std::string path;
};

bool readManifest(const char* path, std::vector<InputFile>* files) {
    std::ifstream stream(path);
    if (!stream) {
        fprintf(stderr, "Unable to open input manifest: %s\n", path);
        return false;
    }
    std::string name;
    std::string file;
    while (stream >> name >> file) {
        if (!name.empty() && name[0] != '#') {
            files->push_back({name, file});
        }
    }
    return !files->empty();
}

bool loadInputs(MNN::Interpreter* interpreter, MNN::Session* session, const char* manifest) {
    std::vector<InputFile> files;
    if (!readManifest(manifest, &files)) {
        return false;
    }
    std::map<std::string, std::string> paths;
    for (const auto& file : files) {
        paths[file.name] = file.path;
    }
    const auto& inputs = interpreter->getSessionInputAll(session);
    if (inputs.size() != paths.size()) {
        fprintf(stderr, "Input count mismatch: model=%zu manifest=%zu\n", inputs.size(), paths.size());
        return false;
    }
    for (const auto& item : inputs) {
        auto pathIt = paths.find(item.first);
        if (pathIt == paths.end() || item.second == nullptr) {
            fprintf(stderr, "Missing input in manifest: %s\n", item.first.c_str());
            return false;
        }
        MNN::Tensor host(item.second, item.second->getDimensionType());
        const size_t bytes = item.second->elementSize() * item.second->getType().bytes();
        std::ifstream input(pathIt->second, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
            fprintf(stderr, "Input size mismatch for %s: expected=%zu file=%s\n", item.first.c_str(), bytes,
                    pathIt->second.c_str());
            return false;
        }
        input.seekg(0);
        input.read(reinterpret_cast<char*>(host.host<void>()), bytes);
        if (!input) {
            return false;
        }
        item.second->copyFromHostTensor(&host);
    }
    return true;
}

float halfToFloat(uint16_t bits) {
    const float sign = (bits & 0x8000U) == 0 ? 1.0f : -1.0f;
    const int exponent = (bits >> 10) & 0x1f;
    const int mantissa = bits & 0x3ff;
    if (exponent == 0) {
        return sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 31) {
        return mantissa == 0 ? sign * INFINITY : NAN;
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exponent - 15);
}

bool writeFloatReference(const MNN::Tensor* output, const char* path) {
    MNN::Tensor host(output, output->getDimensionType());
    if (!output->copyToHostTensor(&host)) {
        return false;
    }
    std::vector<float> values(output->elementSize());
    if (output->getType().code == halide_type_float && output->getType().bits == 32) {
        memcpy(values.data(), host.host<void>(), values.size() * sizeof(float));
    } else if (output->getType().code == halide_type_float && output->getType().bits == 16) {
        const auto* source = static_cast<const uint16_t*>(host.host<void>());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = halfToFloat(source[i]);
        }
    } else {
        fprintf(stderr, "Only float outputs are supported for CPU reference comparison\n");
        return false;
    }
    std::ofstream reference(path, std::ios::binary | std::ios::trunc);
    reference.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    return static_cast<bool>(reference);
}

bool hasRecordedRequest(const char* path) {
    OfflineRpcRequest request;
    return readOfflineRpcRequest(path, &request, false);
}

bool removeStaleRequest(const char* path) {
    std::ifstream existing(path, std::ios::binary);
    if (!existing) {
        return true;
    }
    return std::remove(path) == 0;
}

int run(const char* model, const char* manifest, const char* outputName, const char* reference,
        MNNForwardType backend) {
    std::shared_ptr<MNN::Interpreter> interpreter(MNN::Interpreter::createFromFile(model), MNN::Interpreter::destroy);
    if (!interpreter) {
        fprintf(stderr, "Unable to load MNN model: %s\n", model);
        return 2;
    }
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    MNN::ScheduleConfig config;
    config.type = backend;
    config.backendConfig = &backendConfig;
    MNN::Session* session = interpreter->createSession(config);
    if (!session || !loadInputs(interpreter.get(), session, manifest)) {
        return 3;
    }
    MNN::Tensor* output = interpreter->getSessionOutput(session, outputName);
    if (output == nullptr) {
        fprintf(stderr, "Unable to find output: %s\n", outputName);
        return 4;
    }
    if (backend == MNN_FORWARD_HEXAGON && getenv("MNN_HEXAGON_OFFLINE_RPC_PATH") != nullptr) {
        if (!removeStaleRequest(getenv("MNN_HEXAGON_OFFLINE_RPC_PATH"))) {
            fprintf(stderr, "Unable to remove stale offline RPC request: %s\n",
                    getenv("MNN_HEXAGON_OFFLINE_RPC_PATH"));
            return 5;
        }
    }
    const MNN::ErrorCode code = interpreter->runSession(session);
    if (code != MNN::NO_ERROR) {
        fprintf(stderr, "MNN run failed: backend=%d code=%d\n", backend, code);
        return 5;
    }
    if (backend == MNN_FORWARD_CPU && reference != nullptr && !writeFloatReference(output, reference)) {
        return 6;
    }
    interpreter->releaseSession(session);
    if (backend == MNN_FORWARD_HEXAGON) {
        const char* recordPath = getenv("MNN_HEXAGON_OFFLINE_RPC_PATH");
        if (recordPath == nullptr || !hasRecordedRequest(recordPath)) {
            fprintf(stderr, "Offline RPC request was not written completely: %s\n",
                    recordPath != nullptr ? recordPath : "<unset>");
            return 7;
        }
    }
    printf("mnn_graph_record: backend=%d model=%s output=%s code=%d\n", backend, model, outputName, code);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 6 && strcmp(argv[1], "cpu") == 0) {
        return run(argv[2], argv[3], argv[4], argv[5], MNN_FORWARD_CPU);
    }
    if (argc == 5 && strcmp(argv[1], "record") == 0) {
        return run(argv[2], argv[3], argv[4], nullptr, MNN_FORWARD_HEXAGON);
    }
    fprintf(stderr, "Usage: %s cpu MODEL INPUT_MANIFEST OUTPUT_NAME REFERENCE | record MODEL INPUT_MANIFEST OUTPUT_NAME\n",
            argv[0]);
    return 64;
}
