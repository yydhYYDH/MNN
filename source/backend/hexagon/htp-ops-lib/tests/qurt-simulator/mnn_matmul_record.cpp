#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace {

constexpr int kOutputChannels = 32;
constexpr int kRows           = 4;

float inputValue(int row, int channel) {
  return static_cast<float>((row * 7 + channel * 3) % 17 - 8) / 16.0f;
}

float weightValue(int outputChannel, int inputChannel) {
  return static_cast<float>((outputChannel * 11 + inputChannel * 5) % 19 - 9) / 32.0f;
}

float biasValue(int outputChannel) {
  return static_cast<float>(outputChannel % 9 - 4) / 32.0f;
}

float referenceValue(int row, int outputChannel, int inputChannels) {
  float sum = biasValue(outputChannel);
  for (int inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
    sum += inputValue(row, inputChannel) * weightValue(outputChannel, inputChannel);
  }
  return sum;
}

int createModel(const char *modelPath, int inputChannels) {
  auto input = _Input({ 1, inputChannels, 1, kRows }, NCHW, halide_type_of<float>());
  input->setName("input");
  std::vector<float> weight(kOutputChannels * inputChannels);
  std::vector<float> bias(kOutputChannels);
  for (int outputChannel = 0; outputChannel < kOutputChannels; ++outputChannel) {
    bias[outputChannel] = biasValue(outputChannel);
    for (int inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
      weight[outputChannel * inputChannels + inputChannel] = weightValue(outputChannel, inputChannel);
    }
  }
  auto output =
    _Convert(_Conv(std::move(weight), std::move(bias), input, { inputChannels, kOutputChannels }, { 1, 1 }), NCHW);
  output->setName("output");
  Variable::save({ output }, modelPath);
  return 0;
}

int runModel(const char *modelPath, MNNForwardType backend, const char *outputPath, bool verifyGeneratedModel) {
  std::shared_ptr<Interpreter> interpreter(Interpreter::createFromFile(modelPath), Interpreter::destroy);
  if (!interpreter) {
    fprintf(stderr, "Unable to load MNN model: %s\n", modelPath);
    return 2;
  }
  ScheduleConfig config;
  BackendConfig  backendConfig;
  backendConfig.precision = BackendConfig::Precision_Low;
  config.type             = backend;
  config.backendConfig    = &backendConfig;
  Session *session        = interpreter->createSession(config);
  if (!session) {
    fprintf(stderr, "Unable to create backend session: %d\n", backend);
    return 3;
  }
  Tensor   *input         = interpreter->getSessionInput(session, "input");
  const int inputChannels = input->length(1);
  Tensor    hostInput(input, input->getDimensionType());
  for (int channel = 0; channel < inputChannels; ++channel) {
    for (int row = 0; row < kRows; ++row) {
      hostInput.host<float>()[channel * kRows + row] = inputValue(row, channel);
    }
  }
  input->copyFromHostTensor(&hostInput);
  const ErrorCode code = interpreter->runSession(session);
  if (code != NO_ERROR) {
    fprintf(stderr, "MNN run failed: backend=%d code=%d\n", backend, code);
    return 4;
  }
  if (outputPath != nullptr) {
    Tensor *output = interpreter->getSessionOutput(session, "output");
    Tensor  hostOutput(output, output->getDimensionType());
    output->copyToHostTensor(&hostOutput);
    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(hostOutput.host<float>()), output->elementSize() * sizeof(float));
    if (!file) {
      return 5;
    }
    for (int outputChannel = 0; verifyGeneratedModel && backend == MNN_FORWARD_CPU && outputChannel < kOutputChannels;
         ++outputChannel) {
      for (int row = 0; row < kRows; ++row) {
        const int   index    = outputChannel * kRows + row;
        const float expected = referenceValue(row, outputChannel, inputChannels);
        if (std::fabs(hostOutput.host<float>()[index] - expected) > 1e-6f) {
          fprintf(stderr, "Unexpected CPU reference at %d: actual=%f expected=%f\n", index,
                  hostOutput.host<float>()[index], expected);
          return 6;
        }
      }
    }
  }
  printf("mnn_matmul_record: backend=%d model=%s code=%d\n", backend, modelPath, code);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if ((argc == 3 || argc == 4) && strcmp(argv[1], "create") == 0) {
    const int inputChannels = argc == 4 ? atoi(argv[3]) : 32;
    return inputChannels > 0 && inputChannels % 32 == 0 ? createModel(argv[2], inputChannels) : 65;
  }
  if (argc == 4 && strcmp(argv[1], "cpu") == 0) {
    return runModel(argv[2], MNN_FORWARD_CPU, argv[3], true);
  }
  if (argc == 3 && strcmp(argv[1], "record") == 0) {
    return runModel(argv[2], MNN_FORWARD_HEXAGON, nullptr, false);
  }
  fprintf(stderr, "Usage: %s create MODEL [K] | cpu MODEL REFERENCE | record MODEL\n", argv[0]);
  return 64;
}
