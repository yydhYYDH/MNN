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
#include "../../include/htp_command.h"

namespace {

constexpr uint32_t kSize = 32;

uint16_t floatToFp16(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa = (mantissa | 0x800000U) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa + 0x1000U) >> 13));
}

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

int createAsymmetricRequest(const char *path, const char *referencePath, uint32_t m) {
  constexpr uint32_t k = 64;
  constexpr uint32_t n = 32;
  constexpr uint32_t scaleBlocks = 2;
  const uint32_t activationBytes = m * k * sizeof(uint16_t);
  const uint32_t packedWeightBytes = k * n / 2;
  const uint32_t scaleBytes = n * scaleBlocks * 2 * 2 * sizeof(uint16_t);
  const uint32_t weightBytes = packedWeightBytes + scaleBytes;
  const uint32_t biasBytes = n * sizeof(uint16_t);
  const uint32_t outputBytes = m * 64 * sizeof(uint16_t);
  const OfflineRpcBufferDesc buffers[] = {
    { 201, activationBytes, 2048, 0,                       1 },
    { 202, weightBytes,     2048, 0,                       1 },
    { 203, biasBytes,       256,  0,                       1 },
    { 204, outputBytes,     2048, kOfflineRpcBufferOutput, 1 }
  };

  std::vector<float> activationFloat(m * k);
  std::vector<uint16_t> activation(m * k);
  std::vector<int8_t> logicalWeight(n * k);
  std::vector<uint8_t> rawWeight(n * k / 2);
  std::vector<uint8_t> weight(weightBytes, 0);
  std::vector<uint16_t> bias(n, 0);
  std::vector<uint16_t> result(m * 64, 0);
  std::vector<float> reference(m * 64, 0.0f);
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t ki = 0; ki < k; ++ki) {
      const size_t index = static_cast<size_t>(row) * k + ki;
      activationFloat[index] = (static_cast<int>((row * 7 + ki) % 11) - 5) * 0.125f;
      activation[index] = floatToFp16(activationFloat[index]);
    }
  }
  for (uint32_t o = 0; o < n; ++o) {
    for (uint32_t ki = 0; ki < k; ++ki) {
      const int8_t q = static_cast<int8_t>(((o * 5 + ki * 3 + 1) % 16) - 8);
      logicalWeight[o * k + ki] = q;
    }
    for (uint32_t ki = 0; ki < k; ki += 2) {
      rawWeight[o * (k / 2) + ki / 2] =
        static_cast<uint8_t>(((logicalWeight[o * k + ki] + 8) << 4) |
                             (logicalWeight[o * k + ki + 1] + 8));
    }
  }

  for (uint32_t x = 0; x < k / 32; ++x) {
    uint8_t local[32 * 32] = {};
    uint8_t shuffled[32 * 32] = {};
    for (uint32_t yi = 0; yi < 32; ++yi) {
      const uint8_t *src = rawWeight.data() + yi * (k / 2) + x * 16;
      for (uint32_t xi = 0; xi < 16; ++xi) {
        local[2 * xi * 32 + 2 * yi] = src[xi] >> 4;
        local[2 * xi * 32 + 2 * yi + 1] = src[xi] & 0x0f;
      }
    }
    for (uint32_t q = 0; q < 8; ++q) {
      const uint8_t *src = local + q * 128;
      uint8_t *dst = shuffled + q * 128;
      for (uint32_t i = 0; i < 64; ++i) {
        dst[2 * i] = src[i];
        dst[2 * i + 1] = src[64 + i];
      }
    }
    uint8_t *dst = weight.data() + x * 32 * 16;
    for (uint32_t q = 0; q < 4; ++q) {
      const uint8_t *low = shuffled + q * 256;
      const uint8_t *high = low + 128;
      for (uint32_t i = 0; i < 128; ++i) {
        dst[q * 128 + i] = static_cast<uint8_t>((low[i] & 0x0f) | ((high[i] & 0x0f) << 4));
      }
    }
  }

  uint16_t *scaleData = reinterpret_cast<uint16_t *>(weight.data() + packedWeightBytes);
  for (uint32_t block = 0; block < scaleBlocks; ++block) {
    for (uint32_t o = 0; o < n; ++o) {
      const float scale = (1 + ((o + block) % 3)) * 0.125f;
      const float offset = (static_cast<int>((o + 2 * block) % 5) - 2) * 0.25f;
      scaleData[block * 128 + 2 * o] = floatToFp16(scale);
      scaleData[block * 128 + 2 * o + 1] = floatToFp16(scale);
      scaleData[block * 128 + 64 + 2 * o] = floatToFp16(offset);
      scaleData[block * 128 + 64 + 2 * o + 1] = floatToFp16(offset);
      for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t ki = block * 32; ki < (block + 1) * 32; ++ki) {
          reference[row * 64 + o] += activationFloat[row * k + ki] *
                                     (logicalWeight[o * k + ki] * scale + offset);
        }
      }
    }
  }

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputs = {
    DSPCOMMAND::CreateTensor(builder, 201, 0, activationBytes), DSPCOMMAND::CreateTensor(builder, 202, 0, weightBytes),
    DSPCOMMAND::CreateTensor(builder, 203, 0, biasBytes)
  };
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputs = {
    DSPCOMMAND::CreateTensor(builder, 204, 0, outputBytes)
  };
  const int32_t params[] = { static_cast<int32_t>(m), static_cast<int32_t>(k), static_cast<int32_t>(n),
                             0, 1, 1, 1, 2, static_cast<int32_t>(scaleBlocks), 1 };
  auto command = DSPCOMMAND::CreateCommand(builder, 34, builder.CreateVector(inputs), builder.CreateVector(outputs),
                                           builder.CreateVector(params, 10));
  builder.Finish(command);
  OfflineRpcRequestHeader header = {
    kOfflineRpcRequestMagic, kOfflineRpcVersion, 4, 1, { 0, 204, 0, outputBytes, 0, 0 }
  };
  OfflineRpcCommandDesc commandDesc = { static_cast<uint32_t>(builder.GetSize()), 0 };
  const OfflineRpcChunkDesc chunks[] = {
    { 0, activationBytes }, { 0, weightBytes }, { 0, biasBytes }, { 0, outputBytes }
  };
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output || !writeBytes(output, &header, sizeof(header)) || !writeBytes(output, buffers, sizeof(buffers)) ||
      !writeBytes(output, &commandDesc, sizeof(commandDesc)) ||
      !writeBytes(output, builder.GetBufferPointer(), builder.GetSize()) ||
      !writeBytes(output, &chunks[0], sizeof(chunks[0])) || !writeBytes(output, activation.data(), activationBytes) ||
      !writeBytes(output, &chunks[1], sizeof(chunks[1])) || !writeBytes(output, weight.data(), weightBytes) ||
      !writeBytes(output, &chunks[2], sizeof(chunks[2])) || !writeBytes(output, bias.data(), biasBytes) ||
      !writeBytes(output, &chunks[3], sizeof(chunks[3])) || !writeBytes(output, result.data(), outputBytes)) {
    return 1;
  }
  std::ofstream referenceOutput(referencePath, std::ios::binary | std::ios::trunc);
  if (!referenceOutput || !writeBytes(referenceOutput, reference.data(), reference.size() * sizeof(float))) {
    return 1;
  }
  printf("offline_rpc_host: asymmetric q4 request=%s reference=%s m=%u k=%u n=%u blocks=%u\n",
         path, referencePath, m, k, n, scaleBlocks);
  return 0;
}

int createW8BlockRequest(const char *path, const char *referencePath, uint32_t m, uint32_t scaleBlocks,
                         bool asymmetric) {
  constexpr uint32_t k = 64;
  constexpr uint32_t n = 32;
  const uint32_t activationBytes = m * k * sizeof(uint16_t);
  const uint32_t packedWeightBytes = n * k;
  const uint32_t scaleUnit = asymmetric ? 128 : (scaleBlocks > 1 ? 64 : 32);
  const uint32_t scaleBytes = scaleBlocks * scaleUnit * sizeof(uint16_t);
  const uint32_t weightBytes = packedWeightBytes + scaleBytes;
  const uint32_t biasBytes = n * sizeof(uint16_t);
  const uint32_t outputBytes = m * 64 * sizeof(uint16_t);
  const OfflineRpcBufferDesc buffers[] = {
    { 301, activationBytes, 2048, 0,                       1 },
    { 302, weightBytes,     2048, 0,                       1 },
    { 303, biasBytes,       256,  0,                       1 },
    { 304, outputBytes,     2048, kOfflineRpcBufferOutput, 1 }
  };
  std::vector<float> activationFloat(m * k);
  std::vector<uint16_t> activation(m * k);
  std::vector<int8_t> logicalWeight(n * k);
  std::vector<uint8_t> weight(weightBytes, 0);
  std::vector<uint16_t> bias(n, 0);
  std::vector<uint16_t> result(m * 64, 0);
  std::vector<float> reference(m * 64, 0.0f);
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t ki = 0; ki < k; ++ki) {
      const size_t index = static_cast<size_t>(row) * k + ki;
      activationFloat[index] = (static_cast<int>((row * 7 + ki * 3) % 17) - 8) * 0.0625f;
      activation[index] = floatToFp16(activationFloat[index]);
    }
  }
  for (uint32_t o = 0; o < n; ++o) {
    for (uint32_t ki = 0; ki < k; ++ki) {
      logicalWeight[o * k + ki] = static_cast<int8_t>(((o * 11 + ki * 5 + 3) % 31) - 15);
    }
  }
  for (uint32_t kk = 0; kk < k / 32; ++kk) {
    const size_t blockBase = static_cast<size_t>(kk) * 32 * 32;
    for (uint32_t oy = 0; oy < 32; ++oy) {
      for (uint32_t ix = 0; ix < 32; ++ix) {
        const uint32_t i = kk * 32 + ix;
        const uint32_t ixPair = ix / 2;
        const uint32_t lane = oy * 2 + (ix & 1);
        const size_t dstIndex = blockBase + static_cast<size_t>(ixPair / 2) * 128 + lane * 2 + (ixPair & 1);
        weight[dstIndex] = static_cast<uint8_t>(logicalWeight[oy * k + i]);
      }
    }
  }
  uint16_t *scaleData = reinterpret_cast<uint16_t *>(weight.data() + packedWeightBytes);
  for (uint32_t block = 0; block < scaleBlocks; ++block) {
    for (uint32_t o = 0; o < n; ++o) {
      const float scale = (1 + ((o + block * 2) % 4)) * 0.0625f;
      const float offset = (static_cast<int>((o + block * 3) % 7) - 3) * 0.125f;
      if (scaleBlocks > 1 || asymmetric) {
        scaleData[block * scaleUnit + 2 * o] = floatToFp16(scale);
        scaleData[block * scaleUnit + 2 * o + 1] = floatToFp16(scale);
        if (asymmetric) {
          scaleData[block * scaleUnit + 64 + 2 * o] = floatToFp16(offset);
          scaleData[block * scaleUnit + 64 + 2 * o + 1] = floatToFp16(offset);
        }
      } else {
        scaleData[o] = floatToFp16(scale);
      }
      for (uint32_t row = 0; row < m; ++row) {
        const uint32_t blockSize = k / scaleBlocks;
        for (uint32_t ki = block * blockSize; ki < (block + 1) * blockSize; ++ki) {
          reference[row * 64 + o] +=
            activationFloat[row * k + ki] *
            (logicalWeight[o * k + ki] * scale + (asymmetric ? offset : 0.0f));
        }
      }
    }
  }

  const int32_t params[] = {
    0, 0, 1, 1, 1, 1, 1, 1, 2, 2,
    1, static_cast<int32_t>(m), 1, static_cast<int32_t>(m), static_cast<int32_t>(m * 64), 64,
    64, static_cast<int32_t>(m * 64), 64, 64,
    static_cast<int32_t>(n), 1, 1, 0, 0, 1, static_cast<int32_t>(outputBytes),
    static_cast<int32_t>(scaleBlocks), asymmetric ? 1 : 0
  };
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputs = {
    DSPCOMMAND::CreateTensor(builder, 301, 0, activationBytes), DSPCOMMAND::CreateTensor(builder, 302, 0, weightBytes),
    DSPCOMMAND::CreateTensor(builder, 303, 0, biasBytes)
  };
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputs = {
    DSPCOMMAND::CreateTensor(builder, 304, 0, outputBytes)
  };
  auto command = DSPCOMMAND::CreateCommand(builder, DSP_OP_MATMUL_W8A16_BLOCK_FP16,
                                           builder.CreateVector(inputs), builder.CreateVector(outputs),
                                           builder.CreateVector(params, sizeof(params) / sizeof(params[0])));
  builder.Finish(command);
  OfflineRpcRequestHeader header = {
    kOfflineRpcRequestMagic, kOfflineRpcVersion, 4, 1, { 0, 304, 0, outputBytes, 0, 0 }
  };
  OfflineRpcCommandDesc commandDesc = { static_cast<uint32_t>(builder.GetSize()), 0 };
  const OfflineRpcChunkDesc chunks[] = {
    { 0, activationBytes }, { 0, weightBytes }, { 0, biasBytes }, { 0, outputBytes }
  };
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output || !writeBytes(output, &header, sizeof(header)) || !writeBytes(output, buffers, sizeof(buffers)) ||
      !writeBytes(output, &commandDesc, sizeof(commandDesc)) ||
      !writeBytes(output, builder.GetBufferPointer(), builder.GetSize()) ||
      !writeBytes(output, &chunks[0], sizeof(chunks[0])) || !writeBytes(output, activation.data(), activationBytes) ||
      !writeBytes(output, &chunks[1], sizeof(chunks[1])) || !writeBytes(output, weight.data(), weightBytes) ||
      !writeBytes(output, &chunks[2], sizeof(chunks[2])) || !writeBytes(output, bias.data(), biasBytes) ||
      !writeBytes(output, &chunks[3], sizeof(chunks[3])) || !writeBytes(output, result.data(), outputBytes)) {
    return 1;
  }
  std::ofstream referenceOutput(referencePath, std::ios::binary | std::ios::trunc);
  if (!referenceOutput || !writeBytes(referenceOutput, reference.data(), reference.size() * sizeof(float))) {
    return 1;
  }
  printf("offline_rpc_host: W8 block request=%s reference=%s m=%u k=%u n=%u blocks=%u asymmetric=%d\n",
         path, referencePath, m, k, n, scaleBlocks, asymmetric ? 1 : 0);
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
  if (argc == 4 && strcmp(argv[1], "create-w8-block") == 0) {
    return createW8BlockRequest(argv[2], argv[3], 64, 2, false);
  }
  if (argc == 4 && strcmp(argv[1], "create-w8-block-m1") == 0) {
    return createW8BlockRequest(argv[2], argv[3], 1, 2, false);
  }
  if (argc == 4 && strcmp(argv[1], "create-w8-per-channel") == 0) {
    return createW8BlockRequest(argv[2], argv[3], 64, 1, false);
  }
  if (argc == 4 && strcmp(argv[1], "create-w8-block-asymmetric") == 0) {
    return createW8BlockRequest(argv[2], argv[3], 64, 2, true);
  }
  if (argc == 4 && strcmp(argv[1], "create-w8-block-asymmetric-m1") == 0) {
    return createW8BlockRequest(argv[2], argv[3], 1, 2, true);
  }
  if (argc == 4 && strcmp(argv[1], "create-asymmetric") == 0) {
    return createAsymmetricRequest(argv[2], argv[3], 64);
  }
  if (argc == 4 && strcmp(argv[1], "create-asymmetric-m1") == 0) {
    return createAsymmetricRequest(argv[2], argv[3], 1);
  }
  if (argc == 4 && strcmp(argv[1], "verify-reference") == 0) {
    return verifyReference(argv[2], argv[3]);
  }
  if (argc != 3) {
    fprintf(stderr,
            "Usage: %s create|verify|inspect FILE | create-asymmetric REQUEST REFERENCE | "
            "create-asymmetric-m1 REQUEST REFERENCE | "
            "verify-reference RESPONSE REFERENCE\n",
            argv[0]);
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
