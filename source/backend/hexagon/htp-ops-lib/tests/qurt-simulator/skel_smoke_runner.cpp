#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <remote.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "offline_rpc_protocol.h"

extern "C" {
AEEResult htp_ops_open(const char *uri, remote_handle64 *handle);
AEEResult htp_ops_close(remote_handle64 handle);
AEEResult htp_ops_init_backend(remote_handle64 handle);
AEEResult htp_ops_get_skel_arch(remote_handle64 handle, uint32 *arch);
int       htp_ops_execute_offline_command(const uint8_t *commandData, const int *bufferIds, void *const *bufferPtrs,
                                          int bufferCount);
}

static constexpr int kSize          = 32;
static constexpr int kMeasuredRuns  = 5;
static constexpr int kMaxBuffers    = 64;
static constexpr int kMaxCommands   = 4096;
static uint64        gAverageCycles = 0;
static uint64        gAverageTicks  = 0;
static int           gBadResults    = -1;
static uint16        gFirstResult   = 0;

static bool readExact(FILE *file, void *data, size_t size) {
  return fread(data, 1, size, file) == size;
}

static bool writeExact(FILE *file, const void *data, size_t size) {
  return fwrite(data, 1, size, file) == size;
}

static int runOfflineCommand() {
  FILE                   *request = fopen("offline_rpc_request.bin", "rb");
  OfflineRpcRequestHeader header  = {};
  if (!request || !readExact(request, &header, sizeof(header)) || header.magic != kOfflineRpcRequestMagic ||
      header.version != kOfflineRpcVersion || header.bufferCount == 0 || header.bufferCount > kMaxBuffers ||
      header.commandCount == 0 || header.commandCount > kMaxCommands) {
    if (request) {
      fclose(request);
    }
    return 5;
  }
  OfflineRpcBufferDesc descs[kMaxBuffers] = {};
  if (!readExact(request, descs, header.bufferCount * sizeof(descs[0]))) {
    fclose(request);
    return 6;
  }
  OfflineRpcCommandDesc commandDescs[kMaxCommands] = {};
  if (!readExact(request, commandDescs, header.commandCount * sizeof(commandDescs[0]))) {
    fclose(request);
    return 7;
  }
  printf("offline_rpc_load: buffers=%lu commands=%lu\n", (unsigned long) header.bufferCount,
         (unsigned long) header.commandCount);
  fflush(stdout);
  uint8_t *commands[kMaxCommands] = {};
  int      ids[kMaxBuffers]       = {};
  void    *ptrs[kMaxBuffers]      = {};
  int      outputIndex            = -1;
  int      indicesIndex           = -1;
  int      expectedIndex          = -1;
  uint64   memalignTicks          = 0;
  uint64   memsetTicks            = 0;
  uint64   freadTicks             = 0;
  uint64   logicalBytes           = 0;
  uint64   storedBytes            = 0;
  uint64   chunkCount             = 0;
  for (uint32_t i = 0; i < header.commandCount; ++i) {
    if (commandDescs[i].size == 0 || commandDescs[i].size > 65536) {
      fclose(request);
      return 7;
    }
    commands[i] = (uint8_t *) memalign(128, commandDescs[i].size);
    if (!commands[i] || !readExact(request, commands[i], commandDescs[i].size)) {
      fclose(request);
      for (uint32_t j = 0; j <= i; ++j) {
        free(commands[j]);
      }
      return 7;
    }
  }
  for (uint32_t i = 0; i < header.bufferCount; ++i) {
    printf("offline_rpc_load: buffer=%lu/%lu id=%ld logical_bytes=%lu chunks=%lu\n", (unsigned long) (i + 1),
           (unsigned long) header.bufferCount, (long) descs[i].id, (unsigned long) descs[i].logicalSize,
           (unsigned long) descs[i].chunkCount);
    fflush(stdout);
    ids[i]           = descs[i].id;
    uint64 tickStart = HAP_perf_get_qtimer_count();
    ptrs[i]          = memalign(descs[i].alignment, descs[i].logicalSize);
    memalignTicks += HAP_perf_get_qtimer_count() - tickStart;
    if (!ptrs[i]) {
      fclose(request);
      for (uint32_t j = 0; j <= i; ++j) {
        free(ptrs[j]);
      }
      for (uint32_t j = 0; j < header.commandCount; ++j) {
        free(commands[j]);
      }
      return 8;
    }
    tickStart = HAP_perf_get_qtimer_count();
    memset(ptrs[i], 0, descs[i].logicalSize);
    memsetTicks += HAP_perf_get_qtimer_count() - tickStart;
    logicalBytes += descs[i].logicalSize;
    for (uint32_t chunkIndex = 0; chunkIndex < descs[i].chunkCount; ++chunkIndex) {
      OfflineRpcChunkDesc chunk = {};
      tickStart                 = HAP_perf_get_qtimer_count();
      const bool readDesc       = readExact(request, &chunk, sizeof(chunk));
      freadTicks += HAP_perf_get_qtimer_count() - tickStart;
      if (!readDesc || chunk.offset > descs[i].logicalSize || chunk.size > descs[i].logicalSize - chunk.offset) {
        fclose(request);
        return 8;
      }
      tickStart           = HAP_perf_get_qtimer_count();
      const bool readData = readExact(request, (uint8_t *) ptrs[i] + chunk.offset, chunk.size);
      freadTicks += HAP_perf_get_qtimer_count() - tickStart;
      if (!readData) {
        fclose(request);
        return 8;
      }
      storedBytes += chunk.size;
      ++chunkCount;
    }
    if ((descs[i].flags & kOfflineRpcBufferOutput) &&
        (header.reserved[kOfflineRpcOutputFdIndex] == 0 ||
         descs[i].id == (int32_t) header.reserved[kOfflineRpcOutputFdIndex])) {
      outputIndex = i;
    }
    if (descs[i].id == 203) {
      indicesIndex = i;
    }
    if (descs[i].id == 204) {
      expectedIndex = i;
    }
  }
  printf(
    "offline_rpc_load: complete logical_bytes=%llu stored_bytes=%llu chunks=%llu memalign_ticks=%llu "
    "memset_ticks=%llu fread_ticks=%llu\n",
    (unsigned long long) logicalBytes, (unsigned long long) storedBytes, (unsigned long long) chunkCount,
    (unsigned long long) memalignTicks, (unsigned long long) memsetTicks, (unsigned long long) freadTicks);
  fflush(stdout);
  fclose(request);
  if (outputIndex < 0) {
    for (uint32_t i = 0; i < header.bufferCount; ++i) {
      free(ptrs[i]);
    }
    for (uint32_t i = 0; i < header.commandCount; ++i) {
      free(commands[i]);
    }
    return 9;
  }
  const uint32_t outputOffset = header.reserved[kOfflineRpcOutputOffsetIndex];
  uint32_t       outputBytes  = header.reserved[kOfflineRpcOutputSizeIndex];
  if (outputBytes == 0) {
    outputBytes = descs[outputIndex].logicalSize;
  }
  if (outputOffset > descs[outputIndex].logicalSize || outputBytes > descs[outputIndex].logicalSize - outputOffset) {
    return 9;
  }

  uint64     totalCycles      = 0;
  uint64     totalTicks       = 0;
  int        err              = 0;
  const bool verifyMockMatMul = header.reserved[0] == kOfflineRpcVerifyMockMatMul;
  const bool verifyTopK        = header.reserved[0] == kOfflineRpcVerifyTopK;
  const int  runCount         = verifyMockMatMul ? kMeasuredRuns : 1;
  for (int run = 0; run < runCount; ++run) {
    if (verifyMockMatMul || verifyTopK) {
      memset((uint8_t *) ptrs[outputIndex] + outputOffset, 0, outputBytes);
    }
    uint64 tickStart  = HAP_perf_get_qtimer_count();
    uint64 cycleStart = HAP_perf_get_pcycles();
    for (uint32_t commandIndex = 0; commandIndex < header.commandCount && err == 0; ++commandIndex) {
      err = htp_ops_execute_offline_command(commands[commandIndex], ids, ptrs, header.bufferCount);
      if (err != 0 || (commandIndex + 1) % 64 == 0 || commandIndex + 1 == header.commandCount) {
        printf("offline_rpc_execute: command=%lu/%lu err=%d\n", (unsigned long) (commandIndex + 1),
               (unsigned long) header.commandCount, err);
        fflush(stdout);
      }
    }
    totalCycles += HAP_perf_get_pcycles() - cycleStart;
    totalTicks += HAP_perf_get_qtimer_count() - tickStart;
    if (err != 0) {
      break;
    }
  }
  uint16 *output = (uint16 *) ((uint8_t *) ptrs[outputIndex] + outputOffset);
  int     bad    = 0;
  if (verifyMockMatMul) {
    for (int row = 0; err == 0 && row < kSize; ++row) {
      for (int col = 0; col < kSize; ++col) {
        if (output[row * 64 + col] != 0x5000) {
          ++bad;
        }
      }
    }
  }
  if (verifyTopK) {
    if (expectedIndex < 0 || indicesIndex < 0 || descs[expectedIndex].logicalSize < outputBytes + descs[indicesIndex].logicalSize) {
      ++bad;
    } else {
      const uint8_t *actualValues = (const uint8_t *)output;
      const uint8_t *actualIndices = (const uint8_t *)ptrs[indicesIndex];
      const uint8_t *expected = (const uint8_t *)ptrs[expectedIndex];
      for (uint32_t i = 0; i < outputBytes; ++i) {
        if (actualValues[i] != expected[i]) {
          ++bad;
        }
      }
      for (uint32_t i = 0; i < descs[indicesIndex].logicalSize; ++i) {
        if (actualIndices[i] != expected[outputBytes + i]) {
          ++bad;
        }
      }
    }
  }
  gBadResults                             = bad;
  gFirstResult                            = output[0];
  gAverageCycles                          = totalCycles / runCount;
  gAverageTicks                           = totalTicks / runCount;
  OfflineRpcResponseHeader responseHeader = {
    kOfflineRpcResponseMagic, kOfflineRpcVersion, err, outputBytes, (uint32_t) bad, output[0],
    gAverageCycles,           gAverageTicks
  };
  FILE *response = fopen("offline_rpc_response.bin", "wb");
  bool  wrote    = response && writeExact(response, &responseHeader, sizeof(responseHeader)) &&
               writeExact(response, output, outputBytes);
  if (response) {
    fclose(response);
  }
  for (uint32_t i = 0; i < header.bufferCount; ++i) {
    free(ptrs[i]);
  }
  for (uint32_t i = 0; i < header.commandCount; ++i) {
    free(commands[i]);
  }
  printf("offline_rpc_graph: err=%d buffers=%lu commands=%lu bad=%d first=0x%04lx\n", err,
         (unsigned long) header.bufferCount, (unsigned long) header.commandCount, bad, (unsigned long) gFirstResult);
  return wrote && err == 0 && bad == 0 ? 0 : 10;
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;

  remote_handle64 handle = 0;
  AEEResult       err    = htp_ops_open("file:///libMNN_htpops_skel.so", &handle);
  printf("htp_ops_open: err=%d handle=0x%llx\n", err, (unsigned long long) handle);
  if (err != AEE_SUCCESS) {
    return 1;
  }

  err = htp_ops_init_backend(handle);
  printf("htp_ops_init_backend: err=%d\n", err);
  if (err != AEE_SUCCESS) {
    htp_ops_close(handle);
    return 2;
  }

  uint32 arch = 0;
  err         = htp_ops_get_skel_arch(handle, &arch);
  printf("htp_ops_get_skel_arch: err=%d arch=0x%lx\n", err, (unsigned long) arch);
  if (err != AEE_SUCCESS || arch != 0x79) {
    htp_ops_close(handle);
    return 3;
  }

  const int benchmarkErr = runOfflineCommand();
  printf("offline_rpc timing: pcycles=%llu qtimer_ticks=%llu time_ns=%llu\n", (unsigned long long) gAverageCycles,
         (unsigned long long) gAverageTicks, (unsigned long long) (gAverageTicks * 625 / 12));

  const AEEResult closeErr = htp_ops_close(handle);
  printf("htp_ops_close: err=%d\n", closeErr);
  if (benchmarkErr != 0) {
    return benchmarkErr;
  }
  return closeErr == AEE_SUCCESS ? 0 : 4;
}
