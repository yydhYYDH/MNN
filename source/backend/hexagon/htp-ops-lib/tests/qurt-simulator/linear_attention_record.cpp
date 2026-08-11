#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <string>
#include <vector>

#include "core/TensorUtils.hpp"
#include "schema/current/MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

constexpr int kNumKHeads  = 2;
constexpr int kNumVHeads  = 2;
constexpr int kHeadKDim   = 8;
constexpr int kHeadVDim   = 8;
constexpr int kConvKernel = 4;

bool dumpLogicalOutput(Interpreter *interpreter, Session *session, const std::string &path);

int createModel(const char *path, int sequence) {
  const int keyDim   = kNumKHeads * kHeadKDim;
  const int valueDim = kNumVHeads * kHeadVDim;
  const int convDim  = 2 * keyDim + valueDim;
  auto      qkv      = _Input({ sequence, convDim, 1, 1 }, NC4HW4, halide_type_of<float>());
  auto      gate     = _Input({ sequence, kNumVHeads, 1, 1 }, NC4HW4, halide_type_of<float>());
  auto      beta     = _Input({ sequence, kNumVHeads, 1, 1 }, NC4HW4, halide_type_of<float>());
  auto      convW    = _Input({ convDim, 1, kConvKernel, 1 }, NCHW, halide_type_of<float>());
  qkv->setName("qkv");
  gate->setName("gate");
  beta->setName("beta");
  convW->setName("convW");

  std::shared_ptr<OpT> op(new OpT);
  op->type             = OpType_LinearAttention;
  op->main.type        = OpParameter_LinearAttentionParam;
  op->main.value       = new LinearAttentionParamT;
  auto *param          = op->main.AsLinearAttentionParam();
  param->attn_type     = "gated_delta_rule";
  param->num_k_heads   = kNumKHeads;
  param->num_v_heads   = kNumVHeads;
  param->head_k_dim    = kHeadKDim;
  param->head_v_dim    = kHeadVDim;
  param->use_qk_l2norm = true;
  auto output          = Variable::create(Expr::create(op.get(), { qkv, gate, beta, convW }));
  output->setName("output");
  auto  buffer = Variable::save({ output });
  FILE *file   = fopen(path, "wb");
  if (file == nullptr) {
    return 2;
  }
  const bool ok = fwrite(buffer.data(), 1, buffer.size(), file) == buffer.size();
  fclose(file);
  return ok ? 0 : 3;
}

int createMulModel(const char *path, int elements) {
  VARP input0;
  VARP input1;
  if (elements == 64) {
    input0 = _Input({ 1, 16, 1, 1 }, NC4HW4, halide_type_of<float>());
    input1 = _Input({ 1, 16, 1, 1 }, NC4HW4, halide_type_of<float>());
  } else if (elements == 2048) {
    input0 = _Input({ 16, 128 }, NC4HW4, halide_type_of<float>());
    input1 = _Input({ 16, 128 }, NC4HW4, halide_type_of<float>());
  } else {
    return 8;
  }
  input0->setName("input0");
  input1->setName("input1");
  auto output = _Multiply(input0, input1);
  output->setName("output");
  auto  buffer = Variable::save({ output });
  FILE *file   = fopen(path, "wb");
  if (file == nullptr) {
    return 2;
  }
  const bool ok = fwrite(buffer.data(), 1, buffer.size(), file) == buffer.size();
  fclose(file);
  return ok ? 0 : 3;
}

int createRasterReshapeModel(const char *path) {
  auto input = _Input({ 1, 2048, 1, 1 }, NC4HW4, halide_type_of<float>());
  input->setName("input");
  auto output = _Reshape(input, { 16, 128 });
  output->setName("output");
  auto  buffer = Variable::save({ output });
  FILE *file   = fopen(path, "wb");
  if (file == nullptr) {
    return 2;
  }
  const bool ok = fwrite(buffer.data(), 1, buffer.size(), file) == buffer.size();
  fclose(file);
  return ok ? 0 : 3;
}

int runRasterReshapeModel(const char *path, const char *outputPath, MNNForwardType type) {
  std::shared_ptr<Interpreter> interpreter(Interpreter::createFromFile(path), Interpreter::destroy);
  ScheduleConfig               config;
  BackendConfig                backendConfig;
  backendConfig.precision = type == MNN_FORWARD_CPU ? BackendConfig::Precision_Normal : BackendConfig::Precision_Low;
  config.type             = type;
  config.backendConfig    = &backendConfig;
  Session *session        = interpreter ? interpreter->createSession(config) : nullptr;
  if (session == nullptr) {
    return 5;
  }
  Tensor                 *input = interpreter->getSessionInput(session, "input");
  std::shared_ptr<Tensor> host(Tensor::create<float>(input->shape(), nullptr, Tensor::CAFFE));
  for (int i = 0; i < host->elementSize(); ++i) {
    host->host<float>()[i] = 0.03125f * static_cast<float>((i * 13) % 257 - 128);
  }
  input->copyFromHostTensor(host.get());
  if (interpreter->runSession(session) != NO_ERROR) {
    return 6;
  }
  return dumpLogicalOutput(interpreter.get(), session, outputPath) ? 0 : 7;
}

int runMulModel(const char *path, const char *outputPath, MNNForwardType type) {
  std::shared_ptr<Interpreter> interpreter(Interpreter::createFromFile(path), Interpreter::destroy);
  ScheduleConfig               config;
  BackendConfig                backendConfig;
  backendConfig.precision = type == MNN_FORWARD_CPU ? BackendConfig::Precision_Normal : BackendConfig::Precision_Low;
  config.type             = type;
  config.backendConfig    = &backendConfig;
  Session *session        = interpreter ? interpreter->createSession(config) : nullptr;
  if (session == nullptr) {
    return 5;
  }
  auto inputs = interpreter->getSessionInputAll(session);
  for (const auto &item : inputs) {
    Tensor                 *tensor = item.second;
    std::shared_ptr<Tensor> host(Tensor::create<float>(tensor->shape(), nullptr, Tensor::CAFFE));
    float                  *data = host->host<float>();
    for (int i = 0; i < host->elementSize(); ++i) {
      if (item.first == "input0") {
        data[i] = 0.03125f * static_cast<float>((i * 13) % 31 - 15);
      } else {
        data[i] = 0.0625f * static_cast<float>((i * 7) % 23 - 11);
      }
    }
    tensor->copyFromHostTensor(host.get());
  }
  if (interpreter->runSession(session) != NO_ERROR) {
    return 6;
  }
  return dumpLogicalOutput(interpreter.get(), session, outputPath) ? 0 : 7;
}

void fillInputs(Interpreter *interpreter, Session *session, int tokenOffset) {
  auto inputs = interpreter->getSessionInputAll(session);
  for (const auto &item : inputs) {
    Tensor                 *tensor = item.second;
    std::shared_ptr<Tensor> host(Tensor::create<float>(tensor->shape(), nullptr, Tensor::CAFFE));
    float                  *data          = host->host<float>();
    int                     logicalOffset = 0;
    if (item.first == "qkv") {
      logicalOffset = tokenOffset * (2 * kNumKHeads * kHeadKDim + kNumVHeads * kHeadVDim);
    }
    if (item.first == "gate" || item.first == "beta") {
      logicalOffset = tokenOffset * kNumVHeads;
    }
    for (int i = 0; i < host->elementSize(); ++i) {
      data[i] = 0.01f * static_cast<float>(((logicalOffset + i) * 13) % 17 - 8);
    }
    if (item.first == "gate") {
      for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = -0.1f - 0.01f * ((logicalOffset + i) % 3);
      }
    } else if (item.first == "beta") {
      for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = 0.3f + 0.02f * ((logicalOffset + i) % 4);
      }
    } else if (item.first == "convW") {
      for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = 0.02f * ((i % 7) - 3);
      }
    }
    tensor->copyFromHostTensor(host.get());
  }
}

int runModel(const char *path, bool cpu, const char *dumpPrefix, int outputIndex) {
  std::shared_ptr<Interpreter> interpreter(Interpreter::createFromFile(path), Interpreter::destroy);
  if (!interpreter) {
    return 4;
  }
  ScheduleConfig config;
  BackendConfig  backendConfig;
  backendConfig.precision = cpu ? BackendConfig::Precision_Normal : BackendConfig::Precision_Low;
  config.type             = cpu ? MNN_FORWARD_CPU : MNN_FORWARD_HEXAGON;
  config.backendConfig    = &backendConfig;
  Session *session        = interpreter->createSession(config);
  if (!session) {
    return 5;
  }
  auto inputs = interpreter->getSessionInputAll(session);
  fillInputs(interpreter.get(), session, 0);
  Tensor *output = interpreter->getSessionOutput(session, "output");
  if (cpu) {
    setenv("MNN_LINEAR_ATTENTION_DEBUG_PREFIX", dumpPrefix, 1);
  } else {
    const int         sequence        = inputs.at("qkv")->length(0);
    const int         convDim         = 2 * kNumKHeads * kHeadKDim + kNumVHeads * kHeadVDim;
    const size_t      outputBytes     = ((kHeadVDim + 63) / 64) * sequence * kNumVHeads * 64 * sizeof(uint16_t);
    const size_t      convBytes       = convDim * (kConvKernel - 1) * sizeof(uint16_t);
    const size_t      recurrentBytes  = kNumVHeads * kHeadKDim * kHeadVDim * sizeof(uint16_t);
    const size_t      selectedBytes[] = { outputBytes, convBytes, recurrentBytes };
    const std::string bytes           = std::to_string(selectedBytes[outputIndex]);
    const std::string index           = std::to_string(outputIndex);
    setenv("MNN_HEXAGON_OFFLINE_RPC_OUTPUT_BYTES", bytes.c_str(), 1);
    setenv("MNN_HEXAGON_OFFLINE_RPC_OUTPUT_INDEX", index.c_str(), 1);
  }
  const ErrorCode code = interpreter->runSession(session);
  printf("linear_attention_record: code=%d output_elements=%d\n", code, output->elementSize());
  return code == NO_ERROR ? 0 : 6;
}

bool dumpLogicalOutput(Interpreter *interpreter, Session *session, const std::string &path) {
  Tensor *output = interpreter->getSessionOutput(session, "output");
  Tensor  host(output, Tensor::CAFFE);
  output->copyToHostTensor(&host);
  FILE      *file = fopen(path.c_str(), "wb");
  const bool ok   = file != nullptr && fwrite(host.host<float>(), sizeof(float), host.elementSize(), file) ==
                                       static_cast<size_t>(host.elementSize());
  if (file != nullptr) {
    fclose(file);
  }
  return ok;
}

int runChain(const char *path, const char *dumpPrefix, int decodeSteps, MNNForwardType type) {
  std::shared_ptr<Interpreter> interpreter(Interpreter::createFromFile(path), Interpreter::destroy);
  ScheduleConfig               config;
  BackendConfig                backendConfig;
  backendConfig.precision = type == MNN_FORWARD_CPU ? BackendConfig::Precision_Normal : BackendConfig::Precision_Low;
  config.type             = type;
  config.backendConfig    = &backendConfig;
  Session *session        = interpreter ? interpreter->createSession(config) : nullptr;
  if (session == nullptr) {
    return 5;
  }
  fillInputs(interpreter.get(), session, 0);
  std::string stepPrefix = std::string(dumpPrefix) + ".step0";
  setenv("MNN_LINEAR_ATTENTION_DEBUG_PREFIX", stepPrefix.c_str(), 1);
  if (interpreter->runSession(session) != NO_ERROR) {
    return 6;
  }
  if (!dumpLogicalOutput(interpreter.get(), session, stepPrefix + ".logical.f32")) {
    return 7;
  }

  auto inputs = interpreter->getSessionInputAll(session);
  for (const auto &item : inputs) {
    if (item.first == "convW") {
      continue;
    }
    std::vector<int> shape = item.second->shape();
    shape[0]               = 1;
    interpreter->resizeTensor(item.second, shape);
  }
  interpreter->resizeSession(session);
  for (int step = 0; step < decodeSteps; ++step) {
    fillInputs(interpreter.get(), session, 3 + step);
    stepPrefix = std::string(dumpPrefix) + ".step" + std::to_string(step + 1);
    setenv("MNN_LINEAR_ATTENTION_DEBUG_PREFIX", stepPrefix.c_str(), 1);
    if (interpreter->runSession(session) != NO_ERROR) {
      return 6;
    }
    if (!dumpLogicalOutput(interpreter.get(), session, stepPrefix + ".logical.f32")) {
      return 7;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 4 && strcmp(argv[1], "create") == 0) {
    return createModel(argv[2], atoi(argv[3]));
  }
  if (argc == 4 && strcmp(argv[1], "create-mul") == 0) {
    return createMulModel(argv[2], atoi(argv[3]));
  }
  if (argc == 5 && strcmp(argv[1], "run-mul") == 0) {
    const MNNForwardType type = strcmp(argv[4], "hexagon") == 0 ? MNN_FORWARD_HEXAGON : MNN_FORWARD_CPU;
    return runMulModel(argv[2], argv[3], type);
  }
  if (argc == 3 && strcmp(argv[1], "create-raster-reshape") == 0) {
    return createRasterReshapeModel(argv[2]);
  }
  if (argc == 5 && strcmp(argv[1], "run-raster-reshape") == 0) {
    const MNNForwardType type = strcmp(argv[4], "hexagon") == 0 ? MNN_FORWARD_HEXAGON : MNN_FORWARD_CPU;
    return runRasterReshapeModel(argv[2], argv[3], type);
  }
  if (argc == 4 && strcmp(argv[1], "record") == 0) {
    const int outputIndex = atoi(argv[3]);
    if (outputIndex < 0 || outputIndex > 2) {
      return 64;
    }
    return runModel(argv[2], false, nullptr, outputIndex);
  }
  if (argc == 4 && strcmp(argv[1], "cpu") == 0) {
    return runModel(argv[2], true, argv[3], 0);
  }
  if (argc == 5 && strcmp(argv[1], "cpu-chain") == 0) {
    return runChain(argv[2], argv[3], atoi(argv[4]), MNN_FORWARD_CPU);
  }
  if (argc == 5 && strcmp(argv[1], "hexagon-chain") == 0) {
    return runChain(argv[2], argv[3], atoi(argv[4]), MNN_FORWARD_HEXAGON);
  }
  fprintf(stderr, "Usage: %s create MODEL SEQUENCE | record MODEL OUTPUT_INDEX | cpu MODEL DUMP_PREFIX\n", argv[0]);
  return 64;
}
