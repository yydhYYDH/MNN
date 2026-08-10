#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "offline_rpc_protocol.h"
#include "schema/current/Command_generated.h"

namespace {

constexpr uint32_t kSize = 32;

bool writeBytes(std::ofstream &output, const void *data, size_t size) {
  output.write(static_cast<const char *>(data), size);
  return output.good();
}

int createRequest(const char *path) {
  const uint32_t             activationBytes = kSize * 64 * sizeof(uint16_t);
  const uint32_t             weightBytes     = kSize * kSize / 2 + kSize * sizeof(uint16_t);
  const uint32_t             vectorBytes     = kSize * sizeof(uint16_t);
  const uint32_t             outputBytes     = kSize * 64 * sizeof(uint16_t);
  const OfflineRpcBufferDesc buffers[]       = {
    { 101, activationBytes, 2048, 0,                       1 },
    { 102, weightBytes,     2048, 0,                       1 },
    { 103, vectorBytes,     256,  0,                       1 },
    { 104, outputBytes,     2048, kOfflineRpcBufferOutput, 1 }
  };

  std::vector<uint16_t> activation(kSize * 64, 0);
  std::vector<uint8_t>  weight(weightBytes, 0x99);
  std::vector<uint16_t> bias(kSize, 0);
  std::vector<uint16_t> result(kSize * 64, 0);
  uint16_t             *scales = reinterpret_cast<uint16_t *>(weight.data() + kSize * kSize / 2);
  std::fill(scales, scales + kSize, 0x3c00);
  for (uint32_t row = 0; row < kSize; ++row) {
    for (uint32_t col = 0; col < kSize; ++col) {
      activation[row * 64 + col] = 0x3c00;
    }
  }

  flatbuffers::FlatBufferBuilder                       builder;
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputs = {
    DSPCOMMAND::CreateTensor(builder, 101, 0, activationBytes), DSPCOMMAND::CreateTensor(builder, 102, 0, weightBytes),
    DSPCOMMAND::CreateTensor(builder, 103, 0, vectorBytes)
  };
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputs  = { DSPCOMMAND::CreateTensor(builder, 104, 0,
                                                                                             outputBytes) };
  const int32_t                                        params[] = { 32, 32, 32, 0, 1, 1, 1, 1, 1, 0 };
  auto command = DSPCOMMAND::CreateCommand(builder, 22, builder.CreateVector(inputs), builder.CreateVector(outputs),
                                           builder.CreateVector(params, 10));
  builder.Finish(command);
  OfflineRpcRequestHeader header = {
    kOfflineRpcRequestMagic, kOfflineRpcVersion, 4, 1, { kOfflineRpcVerifyMockMatMul, 104, 0, outputBytes, 0, 0 }
  };
  OfflineRpcCommandDesc     commandDesc = { static_cast<uint32_t>(builder.GetSize()), 0 };
  const OfflineRpcChunkDesc chunks[]    = {
    { 0, activationBytes },
    { 0, weightBytes     },
    { 0, vectorBytes     },
    { 0, outputBytes     }
  };

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output || !writeBytes(output, &header, sizeof(header)) || !writeBytes(output, buffers, sizeof(buffers)) ||
      !writeBytes(output, &commandDesc, sizeof(commandDesc)) ||
      !writeBytes(output, builder.GetBufferPointer(), builder.GetSize()) ||
      !writeBytes(output, &chunks[0], sizeof(chunks[0])) || !writeBytes(output, activation.data(), activationBytes) ||
      !writeBytes(output, &chunks[1], sizeof(chunks[1])) || !writeBytes(output, weight.data(), weightBytes) ||
      !writeBytes(output, &chunks[2], sizeof(chunks[2])) || !writeBytes(output, bias.data(), vectorBytes) ||
      !writeBytes(output, &chunks[3], sizeof(chunks[3])) || !writeBytes(output, result.data(), outputBytes)) {
    fprintf(stderr, "Unable to write offline RPC request: %s\n", path);
    return 1;
  }
  printf("offline_rpc_host: request=%s m=%u k=%u n=%u bytes=%u\n", path, kSize, kSize, kSize,
         static_cast<unsigned>(sizeof(header) + sizeof(buffers) + sizeof(commandDesc) + builder.GetSize() +
                               activationBytes + weightBytes + vectorBytes + outputBytes));
  return 0;
}

int verifyResponse(const char *path) {
  std::ifstream            input(path, std::ios::binary);
  OfflineRpcResponseHeader header = {};
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!input || header.magic != kOfflineRpcResponseMagic || header.version != kOfflineRpcVersion) {
    fprintf(stderr, "Invalid offline RPC response: %s\n", path);
    return 2;
  }
  std::vector<uint16_t> output(header.outputBytes / sizeof(uint16_t));
  input.read(reinterpret_cast<char *>(output.data()), header.outputBytes);
  if (!input || header.status != 0 || header.badResults != 0 || output.size() != kSize * 64 || output[0] != 0x5000) {
    fprintf(stderr, "offline_rpc_host: status=%d bad=%u first=0x%04x output_bytes=%u\n", header.status,
            header.badResults, header.firstResult, header.outputBytes);
    return 3;
  }
  printf("offline_rpc_host: response verified status=%d bad=%u first=0x%04x pcycles=%llu qtimer_ticks=%llu\n",
         header.status, header.badResults, header.firstResult, static_cast<unsigned long long>(header.averageCycles),
         static_cast<unsigned long long>(header.averageTicks));
  return 0;
}

float fp16ToFloat(uint16_t bits) {
  const float sign     = (bits & 0x8000U) ? -1.0f : 1.0f;
  const int   exponent = (bits >> 10) & 0x1f;
  const int   mantissa = bits & 0x3ff;
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa), -24);
  }
  if (exponent == 31) {
    return mantissa == 0 ? sign * INFINITY : NAN;
  }
  return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exponent - 15);
}

int verifyReference(const char *responsePath, const char *referencePath) {
  std::ifstream            response(responsePath, std::ios::binary);
  OfflineRpcResponseHeader header = {};
  response.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!response || header.magic != kOfflineRpcResponseMagic || header.version != kOfflineRpcVersion ||
      header.status != 0 || (header.outputBytes % sizeof(uint16_t)) != 0) {
    return 9;
  }
  std::vector<uint16_t> actual(header.outputBytes / sizeof(uint16_t));
  response.read(reinterpret_cast<char *>(actual.data()), header.outputBytes);
  std::ifstream reference(referencePath, std::ios::binary | std::ios::ate);
  if (!response || !reference || reference.tellg() != static_cast<std::streamoff>(actual.size() * sizeof(float))) {
    return 10;
  }
  reference.seekg(0);
  std::vector<float> expected(actual.size());
  reference.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
  uint32_t    bad              = 0;
  uint32_t    nonFinite        = 0;
  uint32_t    over002          = 0;
  uint32_t    over005          = 0;
  float       maxError         = 0.0f;
  float       maxActual        = 0.0f;
  float       maxExpected      = 0.0f;
  size_t      maxErrorIndex    = 0;
  double      squaredError     = 0.0;
  const char *absToleranceText = getenv("MNN_OFFLINE_RPC_ABS_TOLERANCE");
  const char *rmsToleranceText = getenv("MNN_OFFLINE_RPC_RMS_TOLERANCE");
  const float absTolerance     = absToleranceText != nullptr ? strtof(absToleranceText, nullptr) : 0.01f;
  const float rmsTolerance     = rmsToleranceText != nullptr ? strtof(rmsToleranceText, nullptr) : INFINITY;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float actualValue = fp16ToFloat(actual[i]);
    if (!std::isfinite(actualValue) || !std::isfinite(expected[i])) {
      ++nonFinite;
      ++bad;
      continue;
    }
    const float error = std::fabs(actualValue - expected[i]);
    squaredError += static_cast<double>(error) * error;
    if (error > maxError) {
      maxError      = error;
      maxErrorIndex = i;
      maxActual     = actualValue;
      maxExpected   = expected[i];
    }
    if (error > absTolerance) {
      ++bad;
    }
    if (error > 0.02f) {
      ++over002;
    }
    if (error > 0.05f) {
      ++over005;
    }
  }
  const double rmsError = std::sqrt(squaredError / actual.size());
  printf(
    "offline_rpc_host: CPU reference compared elements=%zu bad=%u abs_tolerance=%g over_0.02=%u over_0.05=%u "
    "non_finite=%u max_error=%g max_index=%zu actual=%g expected=%g rms_error=%g rms_tolerance=%g "
    "pcycles=%llu qtimer_ticks=%llu\n",
    actual.size(), bad, absTolerance, over002, over005, nonFinite, maxError, maxErrorIndex, maxActual, maxExpected,
    rmsError, rmsTolerance, static_cast<unsigned long long>(header.averageCycles),
    static_cast<unsigned long long>(header.averageTicks));
  return bad == 0 && nonFinite == 0 && rmsError <= rmsTolerance ? 0 : 11;
}

int inspectRequest(const char *path) {
  std::ifstream           input(path, std::ios::binary);
  OfflineRpcRequestHeader header = {};
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!input || header.magic != kOfflineRpcRequestMagic || header.version != kOfflineRpcVersion) {
    fprintf(stderr, "Invalid offline RPC request: %s\n", path);
    return 4;
  }
  std::vector<OfflineRpcBufferDesc>  buffers(header.bufferCount);
  std::vector<OfflineRpcCommandDesc> commands(header.commandCount);
  input.read(reinterpret_cast<char *>(buffers.data()), buffers.size() * sizeof(buffers[0]));
  input.read(reinterpret_cast<char *>(commands.data()), commands.size() * sizeof(commands[0]));
  printf("offline_rpc_host: buffers=%u commands=%u output=(fd=%u offset=%u size=%u)\n", header.bufferCount,
         header.commandCount, header.reserved[kOfflineRpcOutputFdIndex], header.reserved[kOfflineRpcOutputOffsetIndex],
         header.reserved[kOfflineRpcOutputSizeIndex]);
  for (uint32_t i = 0; i < header.commandCount; ++i) {
    std::vector<uint8_t> bytes(commands[i].size);
    input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    auto command = flatbuffers::GetRoot<DSPCOMMAND::Command>(bytes.data());
    printf("command[%u]: type=%d inputs=%u outputs=%u params=%u bytes=%u\n", i, command->type(),
           command->inputs() ? command->inputs()->size() : 0, command->outputs() ? command->outputs()->size() : 0,
           command->params() ? command->params()->size() : 0, commands[i].size);
    if (command->inputs()) {
      for (uint32_t tensorIndex = 0; tensorIndex < command->inputs()->size(); ++tensorIndex) {
        const auto tensor = command->inputs()->Get(tensorIndex);
        printf("  input[%u]=(fd=%d offset=%d size=%d)\n", tensorIndex, tensor->fd(), tensor->offset(), tensor->size());
      }
    }
    if (command->outputs()) {
      for (uint32_t tensorIndex = 0; tensorIndex < command->outputs()->size(); ++tensorIndex) {
        const auto tensor = command->outputs()->Get(tensorIndex);
        printf("  output[%u]=(fd=%d offset=%d size=%d)\n", tensorIndex, tensor->fd(), tensor->offset(), tensor->size());
      }
    }
    if (command->params()) {
      printf("  params=");
      for (uint32_t paramIndex = 0; paramIndex < command->params()->size(); ++paramIndex) {
        printf("%s%d", paramIndex == 0 ? "" : ",", command->params()->Get(paramIndex));
      }
      printf("\n");
    }
  }
  uint64_t storedBytes = 0;
  uint64_t chunks      = 0;
  for (uint32_t i = 0; i < header.bufferCount; ++i) {
    printf("buffer[%u]: id=%d logical_size=%u flags=%u chunks=%u\n", i, buffers[i].id, buffers[i].logicalSize,
           buffers[i].flags, buffers[i].chunkCount);
    for (uint32_t chunkIndex = 0; chunkIndex < buffers[i].chunkCount; ++chunkIndex) {
      OfflineRpcChunkDesc chunk = {};
      input.read(reinterpret_cast<char *>(&chunk), sizeof(chunk));
      if (!input || chunk.offset > buffers[i].logicalSize || chunk.size > buffers[i].logicalSize - chunk.offset) {
        return 5;
      }
      input.seekg(chunk.size, std::ios::cur);
      storedBytes += chunk.size;
      ++chunks;
    }
  }
  printf("offline_rpc_host: logical_bytes=%llu stored_bytes=%llu chunks=%llu\n",
         static_cast<unsigned long long>(std::accumulate(
           buffers.begin(), buffers.end(), uint64_t(0),
           [](uint64_t total, const OfflineRpcBufferDesc &buffer) { return total + buffer.logicalSize; })),
         static_cast<unsigned long long>(storedBytes), static_cast<unsigned long long>(chunks));
  return input.good() ? 0 : 5;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 4 && strcmp(argv[1], "verify-reference") == 0) {
    return verifyReference(argv[2], argv[3]);
  }
  if (argc != 3) {
    fprintf(stderr, "Usage: %s create|verify|inspect FILE | verify-reference RESPONSE REFERENCE\n", argv[0]);
    return 64;
  }
  if (strcmp(argv[1], "create") == 0) {
    return createRequest(argv[2]);
  }
  if (strcmp(argv[1], "verify") == 0) {
    return verifyResponse(argv[2]);
  }
  if (strcmp(argv[1], "inspect") == 0) {
    return inspectRequest(argv[2]);
  }
  fprintf(stderr, "Unknown command: %s\n", argv[1]);
  return 64;
}
