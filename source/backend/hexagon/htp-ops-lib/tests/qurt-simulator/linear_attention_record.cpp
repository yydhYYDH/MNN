#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <string>
#include <thread>
#include <vector>

#include "core/TensorUtils.hpp"
#include "half.hpp"
#include "schema/current/MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

constexpr int kNumKHeads  = 16;
constexpr int kNumVHeads  = 16;
constexpr int kHeadKDim   = 128;
constexpr int kHeadVDim   = 128;
constexpr int kConvKernel = 4;

bool dumpLogicalOutput(Interpreter *interpreter, Session *session, const std::string &path);

using Fp16 = half_float::half;

bool writeFp16(const std::string &path, const std::vector<Fp16> &values) {
  FILE      *file = fopen(path.c_str(), "wb");
  const bool ok   = file != nullptr && fwrite(values.data(), sizeof(Fp16), values.size(), file) == values.size();
  if (file != nullptr) {
    fclose(file);
  }
  return ok;
}

float qkvValue(int token, int channel) {
  const int convDim = 2 * kNumKHeads * kHeadKDim + kNumVHeads * kHeadVDim;
  return 0.01f * static_cast<float>(((token * convDim + channel) * 13) % 17 - 8);
}

float gateValue(int token, int head) {
  return -0.1f - 0.01f * ((token * kNumVHeads + head) % 3);
}

float betaValue(int token, int head) {
  return 0.3f + 0.02f * ((token * kNumVHeads + head) % 4);
}

float convWeightValue(int channel, int tap) {
  return 0.02f * (((channel * kConvKernel + tap) % 7) - 3);
}

bool writeStateOracle(const std::string &prefix, int sequence) {
  const int         keyDim        = kNumKHeads * kHeadKDim;
  const int         valueDim      = kNumVHeads * kHeadVDim;
  const int         convDim       = 2 * keyDim + valueDim;
  const int         convStateSize = kConvKernel - 1;
  std::vector<Fp16> convState((size_t) convDim * convStateSize, Fp16(0.0f));
  std::vector<Fp16> recurrent((size_t) kNumVHeads * kHeadKDim * kHeadVDim, Fp16(0.0f));
  std::vector<Fp16> convOutput((size_t) sequence * convDim);

  for (int token = 0; token < sequence; ++token) {
    for (int channel = 0; channel < convDim; ++channel) {
      float sum = 0.0f;
      for (int tap = 0; tap < kConvKernel; ++tap) {
        const float x = tap < convStateSize ? static_cast<float>(convState[(size_t) channel * convStateSize + tap]) :
                                              static_cast<float>(Fp16(qkvValue(token, channel)));
        sum += x * static_cast<float>(Fp16(convWeightValue(channel, tap)));
      }
      convOutput[(size_t) token * convDim + channel] = Fp16(sum / (1.0f + std::exp(-sum)));
      for (int tap = 0; tap + 1 < convStateSize; ++tap) {
        convState[(size_t) channel * convStateSize + tap] = convState[(size_t) channel * convStateSize + tap + 1];
      }
      if (convStateSize > 0) {
        convState[(size_t) channel * convStateSize + convStateSize - 1] = Fp16(qkvValue(token, channel));
      }
    }

    for (int head = 0; head < kNumVHeads; ++head) {
      float q[kHeadKDim];
      float k[kHeadKDim];
      float qNorm = 0.0f;
      float kNorm = 0.0f;
      for (int i = 0; i < kHeadKDim; ++i) {
        q[i] = static_cast<float>(convOutput[(size_t) token * convDim + head * kHeadKDim + i]);
        k[i] = static_cast<float>(convOutput[(size_t) token * convDim + keyDim + head * kHeadKDim + i]);
        qNorm += q[i] * q[i];
        kNorm += k[i] * k[i];
      }
      const float qFactor = 1.0f / std::sqrt(static_cast<float>(kHeadKDim)) / std::sqrt(qNorm + 1.0e-6f);
      const float kFactor = 1.0f / std::sqrt(kNorm + 1.0e-6f);
      for (int i = 0; i < kHeadKDim; ++i) {
        q[i] *= qFactor;
        k[i] *= kFactor;
      }
      const float  decay     = std::exp(gateValue(token, head));
      const float  beta      = betaValue(token, head);
      const size_t stateBase = (size_t) head * kHeadKDim * kHeadVDim;
      for (int j = 0; j < kHeadVDim; ++j) {
        float predicted = 0.0f;
        for (int i = 0; i < kHeadKDim; ++i) {
          predicted += static_cast<float>(recurrent[stateBase + (size_t) i * kHeadVDim + j]) * k[i];
        }
        const float value =
          static_cast<float>(convOutput[(size_t) token * convDim + 2 * keyDim + head * kHeadVDim + j]);
        const float delta = beta * (value - decay * predicted);
        for (int i = 0; i < kHeadKDim; ++i) {
          const size_t index = stateBase + (size_t) i * kHeadVDim + j;
          recurrent[index]   = Fp16(decay * static_cast<float>(recurrent[index]) + k[i] * delta);
        }
      }
    }
  }
  return writeFp16(prefix + ".conv.fp16", convState) && writeFp16(prefix + ".recurrent.fp16", recurrent);
}

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
  if (cpu) {
    setenv("MNN_LINEAR_ATTENTION_DEBUG_PREFIX", dumpPrefix, 1);
  }
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
  if (!cpu) {
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
  if (code != NO_ERROR) {
    return 6;
  }
  if (cpu && !dumpLogicalOutput(interpreter.get(), session, std::string(dumpPrefix) + ".logical.f32")) {
    return 7;
  }
  if (cpu && !writeStateOracle(dumpPrefix, inputs.at("qkv")->length(0))) {
    return 8;
  }
  return 0;
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

int runChunked(const char *path, const char *dumpPrefix, int tokenCount, MNNForwardType type) {
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
    if (item.first == "convW") {
      continue;
    }
    std::vector<int> shape = item.second->shape();
    shape[0]               = 1;
    interpreter->resizeTensor(item.second, shape);
  }
  interpreter->resizeSession(session);
  for (int token = 0; token < tokenCount; ++token) {
    fillInputs(interpreter.get(), session, token);
    const std::string stepPrefix = std::string(dumpPrefix) + ".step" + std::to_string(token);
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

int runTwoSessions(const char *path, const char *dumpPrefix, MNNForwardType type) {
  std::shared_ptr<Interpreter> interpreters[2] = {
    std::shared_ptr<Interpreter>(Interpreter::createFromFile(path), Interpreter::destroy),
    std::shared_ptr<Interpreter>(Interpreter::createFromFile(path), Interpreter::destroy)
  };
  Session       *sessions[2] = { nullptr, nullptr };
  ScheduleConfig config;
  BackendConfig  backendConfig;
  backendConfig.precision = type == MNN_FORWARD_CPU ? BackendConfig::Precision_Normal : BackendConfig::Precision_Low;
  config.type             = type;
  config.backendConfig    = &backendConfig;
  for (int s = 0; s < 2; ++s) {
    sessions[s] = interpreters[s] ? interpreters[s]->createSession(config) : nullptr;
    if (sessions[s] == nullptr) {
      return 5;
    }
    auto inputs = interpreters[s]->getSessionInputAll(sessions[s]);
    for (const auto &item : inputs) {
      if (item.first == "convW") {
        continue;
      }
      std::vector<int> shape = item.second->shape();
      shape[0]               = 1;
      interpreters[s]->resizeTensor(item.second, shape);
    }
    interpreters[s]->resizeSession(sessions[s]);
  }
  for (int round = 0; round < 3; ++round) {
    for (int s = 0; s < 2; ++s) {
      const int tokenOffset = s == 0 ? round : 32 + round;
      fillInputs(interpreters[s].get(), sessions[s], tokenOffset);
      const std::string stepPrefix =
        std::string(dumpPrefix) + ".session" + std::to_string(s) + ".step" + std::to_string(round);
      setenv("MNN_LINEAR_ATTENTION_DEBUG_PREFIX", stepPrefix.c_str(), 1);
      if (interpreters[s]->runSession(sessions[s]) != NO_ERROR) {
        return 6;
      }
      if (!dumpLogicalOutput(interpreters[s].get(), sessions[s], stepPrefix + ".logical.f32")) {
        return 7;
      }
    }
  }
  return 0;
}

int runTwoSessionsConcurrent(const char *path, const char *dumpPrefix, int rounds, MNNForwardType type) {
  if (rounds <= 0) {
    return 64;
  }
  std::shared_ptr<Interpreter> interpreters[2] = {
    std::shared_ptr<Interpreter>(Interpreter::createFromFile(path), Interpreter::destroy),
    std::shared_ptr<Interpreter>(Interpreter::createFromFile(path), Interpreter::destroy)
  };
  Session       *sessions[2] = { nullptr, nullptr };
  ScheduleConfig config;
  BackendConfig  backendConfig;
  backendConfig.precision = type == MNN_FORWARD_CPU ? BackendConfig::Precision_Normal : BackendConfig::Precision_Low;
  config.type             = type;
  config.backendConfig    = &backendConfig;
  for (int s = 0; s < 2; ++s) {
    sessions[s] = interpreters[s] ? interpreters[s]->createSession(config) : nullptr;
    if (sessions[s] == nullptr) {
      return 5;
    }
    auto inputs = interpreters[s]->getSessionInputAll(sessions[s]);
    for (const auto &item : inputs) {
      if (item.first == "convW") {
        continue;
      }
      std::vector<int> shape = item.second->shape();
      shape[0]               = 1;
      interpreters[s]->resizeTensor(item.second, shape);
    }
    interpreters[s]->resizeSession(sessions[s]);
  }

  std::atomic<int>  ready(0);
  std::atomic<bool> start(false);
  int               results[2] = { 0, 0 };
  std::thread       workers[2];
  for (int s = 0; s < 2; ++s) {
    workers[s] = std::thread([&, s]() {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int round = 0; round < rounds && results[s] == 0; ++round) {
        fillInputs(interpreters[s].get(), sessions[s], (s == 0 ? 0 : 32) + round);
        if (interpreters[s]->runSession(sessions[s]) != NO_ERROR) {
          results[s] = 6;
          break;
        }
        const std::string stepPrefix =
          std::string(dumpPrefix) + ".session" + std::to_string(s) + ".step" + std::to_string(round);
        if (!dumpLogicalOutput(interpreters[s].get(), sessions[s], stepPrefix + ".logical.f32")) {
          results[s] = 7;
        }
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  workers[0].join();
  workers[1].join();
  return results[0] != 0 ? results[0] : results[1];
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
  if (argc == 5 && strcmp(argv[1], "cpu-chunked") == 0) {
    return runChunked(argv[2], argv[3], atoi(argv[4]), MNN_FORWARD_CPU);
  }
  if (argc == 5 && strcmp(argv[1], "hexagon-chunked") == 0) {
    return runChunked(argv[2], argv[3], atoi(argv[4]), MNN_FORWARD_HEXAGON);
  }
  if (argc == 4 && strcmp(argv[1], "cpu-two-sessions") == 0) {
    return runTwoSessions(argv[2], argv[3], MNN_FORWARD_CPU);
  }
  if (argc == 4 && strcmp(argv[1], "hexagon-two-sessions") == 0) {
    return runTwoSessions(argv[2], argv[3], MNN_FORWARD_HEXAGON);
  }
  if (argc == 5 && strcmp(argv[1], "cpu-two-sessions-concurrent") == 0) {
    return runTwoSessionsConcurrent(argv[2], argv[3], atoi(argv[4]), MNN_FORWARD_CPU);
  }
  if (argc == 5 && strcmp(argv[1], "hexagon-two-sessions-concurrent") == 0) {
    return runTwoSessionsConcurrent(argv[2], argv[3], atoi(argv[4]), MNN_FORWARD_HEXAGON);
  }
  fprintf(stderr, "Usage: %s create MODEL SEQUENCE | record MODEL OUTPUT_INDEX | cpu MODEL DUMP_PREFIX\n", argv[0]);
  return 64;
}
