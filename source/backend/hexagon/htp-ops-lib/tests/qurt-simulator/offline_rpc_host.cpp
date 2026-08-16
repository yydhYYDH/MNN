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
constexpr uint32_t kTopKRowSize = 4097;
constexpr uint32_t kTopKRows = 2;
constexpr uint32_t kTopK = 40;
constexpr uint32_t kQ4BlockM = 1;
constexpr uint32_t kQ4BlockK = 4096;
constexpr uint32_t kQ4BlockN = 32;
constexpr uint32_t kQ4BlockScaleBlocks = 8;

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

uint16_t floatToFp16(float value);

uint8_t q4BlockNibble(uint32_t outputChannel, uint32_t inputChannel) {
  return static_cast<uint8_t>((outputChannel * 5 + inputChannel * 3) % 16);
}

float q4BlockActivation(uint32_t inputChannel) {
  return static_cast<float>(static_cast<int>((inputChannel * 7) % 13) - 6) / 8.0f;
}

float q4BlockScale(uint32_t outputChannel, uint32_t block) {
  return 0.25f + static_cast<float>((outputChannel + 3 * block) % 5) / 16.0f;
}

float q4BlockBias(uint32_t outputChannel) {
  return static_cast<float>(static_cast<int>(outputChannel % 7) - 3) / 16.0f;
}

void packQ4BlockWeight(std::vector<uint8_t> &packed) {
  const uint32_t inputTiles = kQ4BlockK / 32;
  const uint32_t outputTiles = kQ4BlockN / 32;
  packed.assign(inputTiles * outputTiles * 512, 0);
  uint8_t local[32 * 32] = {};
  uint8_t shuffled[32 * 32] = {};
  for (uint32_t x = 0; x < inputTiles; ++x) {
    memset(local, 8, sizeof(local));
    for (uint32_t yi = 0; yi < 32; ++yi) {
      for (uint32_t xi = 0; xi < 16; ++xi) {
        const uint32_t input = x * 32 + xi * 2;
        const uint8_t  byte = static_cast<uint8_t>((q4BlockNibble(yi, input) << 4) |
                                                   q4BlockNibble(yi, input + 1));
        local[2 * xi * 32 + 2 * yi] = byte >> 4;
        local[2 * xi * 32 + 2 * yi + 1] = byte & 0x0f;
      }
    }
    for (uint32_t q = 0; q < 8; ++q) {
      const uint8_t *src = local + q * 128;
      uint8_t       *dst = shuffled + q * 128;
      for (uint32_t i = 0; i < 64; ++i) {
        dst[2 * i] = src[i];
        dst[2 * i + 1] = src[64 + i];
      }
    }
    uint8_t *dstChunk = packed.data() + x * 512;
    for (uint32_t q = 0; q < 4; ++q) {
      const uint8_t *low = shuffled + q * 256;
      const uint8_t *high = low + 128;
      for (uint32_t i = 0; i < 128; ++i) {
        dstChunk[q * 128 + i] = (low[i] & 0x0f) | ((high[i] & 0x0f) << 4);
      }
    }
  }
}

int createQ4BlockRequest(const char *path, const char *referencePath) {
  const uint32_t weightBytes = (kQ4BlockK / 32) * (kQ4BlockN / 32) * 512;
  const uint32_t scaleBytes = (kQ4BlockN / 32) * kQ4BlockScaleBlocks * 64 * sizeof(uint16_t);
  const uint32_t packedScaleBytes = (kQ4BlockN / 32) * ((kQ4BlockScaleBlocks + 1) / 2) * 64 * sizeof(uint16_t);
  const uint32_t totalWeightBytes = weightBytes + scaleBytes + packedScaleBytes;
  const uint32_t activationBytes = kQ4BlockK * sizeof(uint16_t);
  const uint32_t biasBytes = kQ4BlockN * sizeof(uint16_t);
  const uint32_t outputBytes = kQ4BlockN * sizeof(uint16_t);
  const OfflineRpcBufferDesc buffers[] = {
    {301, activationBytes, 128, 0, 1},
    {302, totalWeightBytes, 128, 0, 1},
    {303, biasBytes, 128, 0, 1},
    {304, outputBytes, 128, kOfflineRpcBufferOutput, 1},
  };

  std::vector<uint16_t> activation(kQ4BlockK);
  std::vector<uint8_t>  weight;
  std::vector<uint16_t> bias(kQ4BlockN);
  std::vector<uint16_t> expected(kQ4BlockN);
  for (uint32_t input = 0; input < kQ4BlockK; ++input) {
    activation[input] = floatToFp16(q4BlockActivation(input));
  }
  packQ4BlockWeight(weight);
  weight.resize(totalWeightBytes, 0);
  uint16_t *scales = reinterpret_cast<uint16_t *>(weight.data() + weightBytes);
  uint16_t *packedScales = reinterpret_cast<uint16_t *>(weight.data() + weightBytes + scaleBytes);
  for (uint32_t block = 0; block < kQ4BlockScaleBlocks; ++block) {
    for (uint32_t output = 0; output < 32; ++output) {
      scales[block * 64 + 2 * output] = floatToFp16(q4BlockScale(output, block));
      scales[block * 64 + 2 * output + 1] = scales[block * 64 + 2 * output];
    }
  }
  for (uint32_t pair = 0; pair < (kQ4BlockScaleBlocks + 1) / 2; ++pair) {
    for (uint32_t output = 0; output < 32; ++output) {
      packedScales[pair * 64 + 2 * output] = scales[pair * 2 * 64 + 2 * output];
      packedScales[pair * 64 + 2 * output + 1] =
          pair * 2 + 1 < kQ4BlockScaleBlocks ? scales[(pair * 2 + 1) * 64 + 2 * output] : 0;
    }
  }
  for (uint32_t output = 0; output < 32; ++output) {
    bias[output] = floatToFp16(q4BlockBias(output));
    float result = q4BlockBias(output);
    for (uint32_t input = 0; input < kQ4BlockK; ++input) {
      const uint32_t block = input / (kQ4BlockK / kQ4BlockScaleBlocks);
      const float    weightValue = static_cast<float>(static_cast<int>(q4BlockNibble(output, input)) - 8);
      result += q4BlockActivation(input) * weightValue * q4BlockScale(output, block);
    }
    expected[output] = floatToFp16(result);
  }

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputs = {
    DSPCOMMAND::CreateTensor(builder, 301, 0, activationBytes),
    DSPCOMMAND::CreateTensor(builder, 302, 0, totalWeightBytes),
    DSPCOMMAND::CreateTensor(builder, 303, 0, biasBytes),
  };
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputs = {
    DSPCOMMAND::CreateTensor(builder, 304, 0, outputBytes),
  };
  const int32_t params[] = {static_cast<int32_t>(kQ4BlockM), static_cast<int32_t>(kQ4BlockK),
                            static_cast<int32_t>(kQ4BlockN), 0, 1, 1, 1, static_cast<int32_t>(kQ4BlockK / 32),
                            static_cast<int32_t>(kQ4BlockScaleBlocks), 0, 0};
  auto command = DSPCOMMAND::CreateCommand(builder, 34, builder.CreateVector(inputs), builder.CreateVector(outputs),
                                           builder.CreateVector(params, 11));
  builder.Finish(command);
  OfflineRpcRequestHeader header = {
    kOfflineRpcRequestMagic, kOfflineRpcVersion, 4, 1, {0, 304, 0, outputBytes, 0, 0}
  };
  OfflineRpcCommandDesc commandDesc = {static_cast<uint32_t>(builder.GetSize()), 0};
  const OfflineRpcChunkDesc chunks[] = {{0, activationBytes}, {0, totalWeightBytes}, {0, biasBytes}, {0, outputBytes}};
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  std::ofstream reference(referencePath, std::ios::binary | std::ios::trunc);
  if (!output || !reference || !writeBytes(output, &header, sizeof(header)) ||
      !writeBytes(output, buffers, sizeof(buffers)) || !writeBytes(output, &commandDesc, sizeof(commandDesc)) ||
      !writeBytes(output, builder.GetBufferPointer(), builder.GetSize()) ||
      !writeBytes(output, &chunks[0], sizeof(chunks[0])) || !writeBytes(output, activation.data(), activationBytes) ||
      !writeBytes(output, &chunks[1], sizeof(chunks[1])) || !writeBytes(output, weight.data(), totalWeightBytes) ||
      !writeBytes(output, &chunks[2], sizeof(chunks[2])) || !writeBytes(output, bias.data(), biasBytes) ||
      !writeBytes(output, &chunks[3], sizeof(chunks[3])) || !writeBytes(output, expected.data(), outputBytes) ||
      !writeBytes(reference, expected.data(), outputBytes)) {
    fprintf(stderr, "Unable to write Q4 block oracle: %s\n", path);
    return 1;
  }
  printf("offline_rpc_host: Q4 block request=%s m=%u k=%u n=%u scale_blocks=%u\n", path, kQ4BlockM, kQ4BlockK,
         kQ4BlockN, kQ4BlockScaleBlocks);
  return 0;
}

int extractCommand(const char *inputPath, int commandIndex, const char *outputPath) {
  std::ifstream input(inputPath, std::ios::binary);
  OfflineRpcRequestHeader header = {};
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!input || header.magic != kOfflineRpcRequestMagic || commandIndex < 0 ||
      static_cast<uint32_t>(commandIndex) >= header.commandCount) {
    return 2;
  }
  std::vector<OfflineRpcBufferDesc> buffers(header.bufferCount);
  std::vector<OfflineRpcCommandDesc> commandDescs(header.commandCount);
  input.read(reinterpret_cast<char *>(buffers.data()), buffers.size() * sizeof(buffers[0]));
  input.read(reinterpret_cast<char *>(commandDescs.data()), commandDescs.size() * sizeof(commandDescs[0]));
  if (!input) {
    return 3;
  }
  std::vector<std::vector<uint8_t>> commands(header.commandCount);
  for (uint32_t i = 0; i < header.commandCount; ++i) {
    commands[i].resize(commandDescs[i].size);
    input.read(reinterpret_cast<char *>(commands[i].data()), commands[i].size());
  }
  struct BufferData {
    OfflineRpcBufferDesc desc;
    std::vector<uint8_t> data;
  };
  std::vector<BufferData> bufferData(header.bufferCount);
  for (uint32_t i = 0; i < header.bufferCount; ++i) {
    bufferData[i].desc = buffers[i];
    bufferData[i].data.resize(buffers[i].logicalSize);
    std::vector<OfflineRpcChunkDesc> chunks(buffers[i].chunkCount);
    for (uint32_t j = 0; j < buffers[i].chunkCount; ++j) {
      input.read(reinterpret_cast<char *>(&chunks[j]), sizeof(chunks[j]));
      const OfflineRpcChunkDesc &chunk = chunks[j];
      if (!input || chunk.offset > buffers[i].logicalSize || chunk.size > buffers[i].logicalSize - chunk.offset) {
        return 4;
      }
    }
    for (const OfflineRpcChunkDesc &chunk : chunks) {
      input.read(reinterpret_cast<char *>(bufferData[i].data.data() + chunk.offset), chunk.size);
      if (!input) {
        return 4;
      }
    }
  }
  const auto *command = flatbuffers::GetRoot<DSPCOMMAND::Command>(commands[commandIndex].data());
  if (command == nullptr || command->outputs() == nullptr || command->outputs()->size() == 0) {
    return 5;
  }
  std::vector<int32_t> selectedIds;
  auto addTensor = [&selectedIds](const DSPCOMMAND::Tensor *tensor) {
    if (tensor == nullptr || tensor->fd() < 0 ||
        std::find(selectedIds.begin(), selectedIds.end(), tensor->fd()) != selectedIds.end()) {
      return;
    }
    selectedIds.push_back(tensor->fd());
  };
  if (command->inputs() != nullptr) {
    for (unsigned int i = 0; i < command->inputs()->size(); ++i) {
      addTensor(command->inputs()->Get(i));
    }
  }
  for (unsigned int i = 0; i < command->outputs()->size(); ++i) {
    addTensor(command->outputs()->Get(i));
  }
  std::vector<BufferData> selected;
  for (int32_t id : selectedIds) {
    auto iter = std::find_if(bufferData.begin(), bufferData.end(), [id](const BufferData &buffer) {
      return buffer.desc.id == id;
    });
    if (iter == bufferData.end()) {
      return 6;
    }
    selected.push_back(*iter);
  }
  const auto *outputTensor = command->outputs()->Get(0);
  const int outputFd = outputTensor->fd();
  const int outputOffset = outputTensor->offset();
  const int outputBytes = command->params() != nullptr && command->params()->size() > 2
                              ? command->params()->Get(2) * static_cast<int>(sizeof(uint16_t))
                              : 0;
  OfflineRpcRequestHeader outputHeader = {
    kOfflineRpcRequestMagic, kOfflineRpcVersion, static_cast<uint32_t>(selected.size()), 1,
    {0, static_cast<uint32_t>(outputFd), static_cast<uint32_t>(outputOffset), static_cast<uint32_t>(outputBytes), 0, 0}
  };
  OfflineRpcCommandDesc commandDesc = {static_cast<uint32_t>(commands[commandIndex].size()), 0};
  std::vector<OfflineRpcBufferDesc> selectedDescs;
  selectedDescs.reserve(selected.size());
  for (const auto &buffer : selected) {
    OfflineRpcBufferDesc desc = buffer.desc;
    desc.chunkCount           = 1;
    selectedDescs.push_back(desc);
  }
  std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
  if (!output || !writeBytes(output, &outputHeader, sizeof(outputHeader)) ||
      !writeBytes(output, selectedDescs.data(), selectedDescs.size() * sizeof(selectedDescs[0])) ||
      !writeBytes(output, &commandDesc, sizeof(commandDesc)) ||
      !writeBytes(output, commands[commandIndex].data(), commands[commandIndex].size())) {
    return 7;
  }
  for (const auto &buffer : selected) {
    OfflineRpcChunkDesc chunk = {0, buffer.desc.logicalSize};
    if (!writeBytes(output, &chunk, sizeof(chunk)) || !writeBytes(output, buffer.data.data(), buffer.data.size())) {
      return 8;
    }
  }
  printf("offline_rpc_host: extracted command=%d type=%u buffers=%zu output_fd=%d offset=%d bytes=%d\n", commandIndex,
         command->type(), selected.size(), outputFd, outputOffset, outputBytes);
  return 0;
}

int extractQ4Compact(const char *inputPath, int commandIndex, const char *outputPath) {
  std::ifstream input(inputPath, std::ios::binary);
  OfflineRpcRequestHeader header = {};
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!input || header.magic != kOfflineRpcRequestMagic || commandIndex < 0 ||
      static_cast<uint32_t>(commandIndex) >= header.commandCount) {
    return 20;
  }
  std::vector<OfflineRpcBufferDesc> buffers(header.bufferCount);
  std::vector<OfflineRpcCommandDesc> commandDescs(header.commandCount);
  input.read(reinterpret_cast<char *>(buffers.data()), buffers.size() * sizeof(buffers[0]));
  input.read(reinterpret_cast<char *>(commandDescs.data()), commandDescs.size() * sizeof(commandDescs[0]));
  if (!input) {
    return 21;
  }
  std::vector<std::vector<uint8_t>> commands(header.commandCount);
  for (uint32_t i = 0; i < header.commandCount; ++i) {
    commands[i].resize(commandDescs[i].size);
    input.read(reinterpret_cast<char *>(commands[i].data()), commands[i].size());
  }
  struct BufferData {
    OfflineRpcBufferDesc desc;
    std::vector<uint8_t> data;
  };
  std::vector<BufferData> bufferData(header.bufferCount);
  for (uint32_t i = 0; i < header.bufferCount; ++i) {
    bufferData[i].desc = buffers[i];
    bufferData[i].data.resize(buffers[i].logicalSize);
    std::vector<OfflineRpcChunkDesc> chunks(buffers[i].chunkCount);
    for (uint32_t j = 0; j < buffers[i].chunkCount; ++j) {
      input.read(reinterpret_cast<char *>(&chunks[j]), sizeof(chunks[j]));
      if (!input || chunks[j].offset > buffers[i].logicalSize ||
          chunks[j].size > buffers[i].logicalSize - chunks[j].offset) {
        return 22;
      }
    }
    for (const auto &chunk : chunks) {
      input.read(reinterpret_cast<char *>(bufferData[i].data.data() + chunk.offset), chunk.size);
      if (!input) {
        return 22;
      }
    }
  }
  const auto *source = flatbuffers::GetRoot<DSPCOMMAND::Command>(commands[commandIndex].data());
  if (source == nullptr || source->type() != 34 || source->inputs() == nullptr || source->inputs()->size() < 2 ||
      source->outputs() == nullptr || source->outputs()->size() == 0 || source->params() == nullptr ||
      source->params()->size() < 9) {
    return 23;
  }
  const int32_t m = source->params()->Get(0);
  const int32_t k = source->params()->Get(1);
  const int32_t n = source->params()->Get(2);
  const int32_t scaleBlocks = source->params()->Get(8);
  if (m <= 0 || k <= 0 || n <= 0 || scaleBlocks <= 0) {
    return 24;
  }
  const size_t activationBytes = static_cast<size_t>(m) * k * sizeof(uint16_t);
  const size_t outputBytes = static_cast<size_t>(m) * n * sizeof(uint16_t);
  const size_t icP = (static_cast<size_t>(k) + 31) / 32;
  const size_t ocP = (static_cast<size_t>(n) + 31) / 32;
  const size_t weightBytes = icP * ocP * 32 * 16;
  const size_t scaleBytes = ocP * static_cast<size_t>(scaleBlocks) * 64 * sizeof(uint16_t);
  const size_t packedScaleBytes = ocP * ((static_cast<size_t>(scaleBlocks) + 1) / 2) * 64 * sizeof(uint16_t);
  const size_t packedWeightBytes = weightBytes + scaleBytes + packedScaleBytes;
  auto findBuffer = [&bufferData](int32_t id) -> const BufferData * {
    for (const auto &buffer : bufferData) {
      if (buffer.desc.id == id) {
        return &buffer;
      }
    }
    return nullptr;
  };
  auto copyTensor = [&findBuffer](const DSPCOMMAND::Tensor *tensor, size_t bytes, std::vector<uint8_t> *dst) -> bool {
    if (tensor == nullptr || tensor->fd() < 0 || tensor->offset() < 0) {
      return false;
    }
    const BufferData *buffer = findBuffer(tensor->fd());
    const size_t offset = static_cast<size_t>(tensor->offset());
    if (buffer == nullptr || offset > buffer->data.size() || bytes > buffer->data.size() - offset) {
      return false;
    }
    dst->assign(buffer->data.begin() + offset, buffer->data.begin() + offset + bytes);
    return true;
  };
  std::vector<uint8_t> activation;
  std::vector<uint8_t> weight;
  if (!copyTensor(source->inputs()->Get(0), activationBytes, &activation) ||
      !copyTensor(source->inputs()->Get(1), packedWeightBytes, &weight)) {
    return 25;
  }
  const int32_t activationFd = 401;
  const int32_t weightFd = 402;
  const int32_t outputFd = 403;
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputs = {
    DSPCOMMAND::CreateTensor(builder, activationFd, 0, static_cast<int32_t>(activationBytes)),
    DSPCOMMAND::CreateTensor(builder, weightFd, 0, static_cast<int32_t>(packedWeightBytes)),
    DSPCOMMAND::CreateTensor(builder, -1, 0, 0)
  };
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputs = {
    DSPCOMMAND::CreateTensor(builder, outputFd, 0, static_cast<int32_t>(outputBytes))
  };
  std::vector<int32_t> params;
  params.reserve(source->params()->size());
  for (unsigned int i = 0; i < source->params()->size(); ++i) {
    params.push_back(source->params()->Get(i));
  }
  auto command = DSPCOMMAND::CreateCommand(builder, source->type(), builder.CreateVector(inputs),
                                            builder.CreateVector(outputs), builder.CreateVector(params));
  builder.Finish(command);
  std::vector<uint8_t> output(outputBytes, 0);
  const OfflineRpcBufferDesc outputBuffers[] = {
    {activationFd, static_cast<uint32_t>(activation.size()), 128, 0, 1},
    {weightFd, static_cast<uint32_t>(weight.size()), 128, 0, 1},
    {outputFd, static_cast<uint32_t>(output.size()), 128, kOfflineRpcBufferOutput, 1}
  };
  const OfflineRpcRequestHeader outputHeader = {
    kOfflineRpcRequestMagic, kOfflineRpcVersion, 3, 1,
    {0, static_cast<uint32_t>(outputFd), 0, static_cast<uint32_t>(outputBytes), 0, 0}
  };
  const OfflineRpcCommandDesc outputCommand = {static_cast<uint32_t>(builder.GetSize()), 0};
  const OfflineRpcChunkDesc activationChunk = {0, static_cast<uint32_t>(activation.size())};
  const OfflineRpcChunkDesc weightChunk = {0, static_cast<uint32_t>(weight.size())};
  const OfflineRpcChunkDesc outputChunk = {0, static_cast<uint32_t>(output.size())};
  std::ofstream outputFile(outputPath, std::ios::binary | std::ios::trunc);
  if (!outputFile || !writeBytes(outputFile, &outputHeader, sizeof(outputHeader)) ||
      !writeBytes(outputFile, outputBuffers, sizeof(outputBuffers)) ||
      !writeBytes(outputFile, &outputCommand, sizeof(outputCommand)) ||
      !writeBytes(outputFile, builder.GetBufferPointer(), builder.GetSize()) ||
      !writeBytes(outputFile, &activationChunk, sizeof(activationChunk)) ||
      !writeBytes(outputFile, activation.data(), activation.size()) ||
      !writeBytes(outputFile, &weightChunk, sizeof(weightChunk)) ||
      !writeBytes(outputFile, weight.data(), weight.size()) ||
      !writeBytes(outputFile, &outputChunk, sizeof(outputChunk)) ||
      !writeBytes(outputFile, output.data(), output.size())) {
    return 26;
  }
  printf("offline_rpc_host: compact Q4 command=%d m=%d k=%d n=%d scale_blocks=%d bytes=%zu\n", commandIndex, m, k,
         n, scaleBlocks, activation.size() + weight.size() + output.size());
  return 0;
}

uint16_t floatToFp16(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
  const uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) return static_cast<uint16_t>(sign);
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

int createTopKRequest(const char *path) {
  const uint32_t inputBytes = kTopKRowSize * kTopKRows * sizeof(uint16_t);
  const uint32_t valueBytes = kTopK * kTopKRows * sizeof(uint16_t);
  const uint32_t indexBytes = kTopK * kTopKRows * sizeof(int32_t);
  const uint32_t expectedBytes = valueBytes + indexBytes;
  const OfflineRpcBufferDesc buffers[] = {
    {201, inputBytes, 128, 0, 1}, {202, valueBytes, 128, kOfflineRpcBufferOutput, 1},
    {203, indexBytes, 128, kOfflineRpcBufferOutput, 1}, {204, expectedBytes, 128, 0, 1}
  };
  std::vector<uint16_t> input(kTopKRowSize * kTopKRows);
  std::vector<uint16_t> expectedValues(kTopK * kTopKRows);
  std::vector<int32_t> expectedIndices(kTopK * kTopKRows);
  for (uint32_t row = 0; row < kTopKRows; ++row) {
    std::vector<std::pair<float, int32_t>> ranked;
    ranked.reserve(kTopKRowSize);
    for (uint32_t col = 0; col < kTopKRowSize; ++col) {
      const float value = static_cast<float>((col * 17 + row * 13) % 101) / 10.0f - 5.0f;
      input[row * kTopKRowSize + col] = floatToFp16(value);
      ranked.emplace_back(value, static_cast<int32_t>(col));
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
      if (a.first != b.first) return a.first > b.first;
      return a.second < b.second;
    });
    for (uint32_t i = 0; i < kTopK; ++i) {
      expectedValues[row * kTopK + i] = floatToFp16(ranked[i].first);
      expectedIndices[row * kTopK + i] = ranked[i].second;
    }
  }
  std::vector<uint8_t> expected(expectedBytes, 0);
  memcpy(expected.data(), expectedValues.data(), valueBytes);
  memcpy(expected.data() + valueBytes, expectedIndices.data(), indexBytes);
  std::vector<uint16_t> outputValues(kTopK * kTopKRows, 0);
  std::vector<int32_t> outputIndices(kTopK * kTopKRows, 0);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputs = {
    DSPCOMMAND::CreateTensor(builder, 201, 0, inputBytes)
  };
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputs = {
    DSPCOMMAND::CreateTensor(builder, 202, 0, valueBytes),
    DSPCOMMAND::CreateTensor(builder, 203, 0, indexBytes)
  };
  const int32_t params[] = {static_cast<int32_t>(kTopKRowSize), static_cast<int32_t>(kTopKRows),
                            static_cast<int32_t>(kTopK), 2};
  auto command = DSPCOMMAND::CreateCommand(builder, 53, builder.CreateVector(inputs), builder.CreateVector(outputs),
                                           builder.CreateVector(params, 4));
  builder.Finish(command);
  OfflineRpcRequestHeader header = {
    kOfflineRpcRequestMagic, kOfflineRpcVersion, 4, 1,
    {kOfflineRpcVerifyTopK, 202, 0, valueBytes, 203, indexBytes}
  };
  OfflineRpcCommandDesc commandDesc = {static_cast<uint32_t>(builder.GetSize()), 0};
  const OfflineRpcChunkDesc chunks[] = {{0, inputBytes}, {0, valueBytes}, {0, indexBytes}, {0, expectedBytes}};
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output || !writeBytes(output, &header, sizeof(header)) || !writeBytes(output, buffers, sizeof(buffers)) ||
      !writeBytes(output, &commandDesc, sizeof(commandDesc)) ||
      !writeBytes(output, builder.GetBufferPointer(), builder.GetSize()) ||
      !writeBytes(output, &chunks[0], sizeof(chunks[0])) || !writeBytes(output, input.data(), inputBytes) ||
      !writeBytes(output, &chunks[1], sizeof(chunks[1])) || !writeBytes(output, outputValues.data(), valueBytes) ||
      !writeBytes(output, &chunks[2], sizeof(chunks[2])) || !writeBytes(output, outputIndices.data(), indexBytes) ||
      !writeBytes(output, &chunks[3], sizeof(chunks[3])) || !writeBytes(output, expected.data(), expectedBytes)) {
    fprintf(stderr, "Unable to write TopK offline RPC request: %s\n", path);
    return 1;
  }
  printf("offline_rpc_host: TopK request=%s rowSize=%u rows=%u k=%u\n", path, kTopKRowSize, kTopKRows, kTopK);
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

int verifyFp16Reference(const char *responsePath, const char *referencePath, bool unpackQwen35State) {
  std::ifstream            response(responsePath, std::ios::binary);
  OfflineRpcResponseHeader header = {};
  response.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!response || header.magic != kOfflineRpcResponseMagic || header.version != kOfflineRpcVersion ||
      header.status != 0 || (header.outputBytes % sizeof(uint16_t)) != 0) {
    return 12;
  }
  std::vector<uint16_t> actual(header.outputBytes / sizeof(uint16_t));
  response.read(reinterpret_cast<char *>(actual.data()), header.outputBytes);
  std::ifstream reference(referencePath, std::ios::binary | std::ios::ate);
  if (!response || !reference || reference.tellg() != static_cast<std::streamoff>(header.outputBytes)) {
    return 13;
  }
  reference.seekg(0);
  std::vector<uint16_t> expected(actual.size());
  reference.read(reinterpret_cast<char *>(expected.data()), header.outputBytes);
  if (unpackQwen35State) {
    constexpr int heads  = 16;
    constexpr int rows   = 128;
    constexpr int cols   = 128;
    constexpr int kTiles = rows / 32;
    if (actual.size() != (size_t) heads * rows * cols) {
      return 15;
    }
    std::vector<uint16_t> logical(actual.size());
    for (int head = 0; head < heads; ++head) {
      const size_t headBase = (size_t) head * rows * cols;
      for (int row = 0; row < rows; ++row) {
        const int kt = row / 32;
        for (int col = 0; col < cols; ++col) {
          const int    nt     = col / 32;
          const size_t packed = ((size_t) nt * kTiles + kt) * 1024 + ((row & 31) / 2) * 64 + (col & 31) * 2 + (row & 1);
          logical[headBase + (size_t) row * cols + col] = actual[headBase + packed];
        }
      }
    }
    actual.swap(logical);
  }
  uint32_t    bad              = 0;
  uint32_t    nonFinite        = 0;
  float       maxError         = 0.0f;
  size_t      maxErrorIndex    = 0;
  double      squaredError     = 0.0;
  double      squaredExpected  = 0.0;
  double      dot              = 0.0;
  double      squaredActual    = 0.0;
  const char *absToleranceText = getenv("MNN_OFFLINE_RPC_ABS_TOLERANCE");
  const char *rmsToleranceText = getenv("MNN_OFFLINE_RPC_RMS_TOLERANCE");
  const float absTolerance     = absToleranceText != nullptr ? strtof(absToleranceText, nullptr) : 0.001f;
  const float rmsTolerance     = rmsToleranceText != nullptr ? strtof(rmsToleranceText, nullptr) : INFINITY;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float actualValue   = fp16ToFloat(actual[i]);
    const float expectedValue = fp16ToFloat(expected[i]);
    if (!std::isfinite(actualValue) || !std::isfinite(expectedValue)) {
      ++nonFinite;
      ++bad;
      continue;
    }
    const float error = std::fabs(actualValue - expectedValue);
    squaredError += static_cast<double>(error) * error;
    squaredExpected += static_cast<double>(expectedValue) * expectedValue;
    squaredActual += static_cast<double>(actualValue) * actualValue;
    dot += static_cast<double>(actualValue) * expectedValue;
    if (error > maxError) {
      maxError      = error;
      maxErrorIndex = i;
    }
    if (error > absTolerance) {
      ++bad;
    }
  }
  const double rmsError = std::sqrt(squaredError / actual.size());
  const double nrmse    = squaredExpected > 0.0 ? std::sqrt(squaredError / squaredExpected) : 0.0;
  const double cosine =
    squaredActual > 0.0 && squaredExpected > 0.0 ? dot / std::sqrt(squaredActual * squaredExpected) : 1.0;
  printf(
    "offline_rpc_host: FP16 reference compared elements=%zu bad=%u abs_tolerance=%g non_finite=%u "
    "max_error=%g max_index=%zu rms_error=%g rms_tolerance=%g nrmse=%g cosine=%.12g "
    "pcycles=%llu qtimer_ticks=%llu\n",
    actual.size(), bad, absTolerance, nonFinite, maxError, maxErrorIndex, rmsError, rmsTolerance, nrmse, cosine,
    static_cast<unsigned long long>(header.averageCycles), static_cast<unsigned long long>(header.averageTicks));
  return bad == 0 && nonFinite == 0 && rmsError <= rmsTolerance ? 0 : 14;
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
  if (argc == 4 && strcmp(argv[1], "create-q4block") == 0) {
    return createQ4BlockRequest(argv[2], argv[3]);
  }
  if (argc == 5 && strcmp(argv[1], "extract-command") == 0) {
    return extractCommand(argv[2], atoi(argv[3]), argv[4]);
  }
  if (argc == 5 && strcmp(argv[1], "extract-q4compact") == 0) {
    return extractQ4Compact(argv[2], atoi(argv[3]), argv[4]);
  }
  if (argc == 3 && strcmp(argv[1], "create-topk") == 0) {
    return createTopKRequest(argv[2]);
  }
  if (argc == 4 && strcmp(argv[1], "verify-reference") == 0) {
    return verifyReference(argv[2], argv[3]);
  }
  if (argc == 4 && strcmp(argv[1], "verify-fp16-reference") == 0) {
    return verifyFp16Reference(argv[2], argv[3], false);
  }
  if (argc == 4 && strcmp(argv[1], "verify-qwen35-packed-state") == 0) {
    return verifyFp16Reference(argv[2], argv[3], true);
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
