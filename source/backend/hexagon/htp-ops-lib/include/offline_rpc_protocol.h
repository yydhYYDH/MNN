#pragma once

#include <stdint.h>

static constexpr uint32_t kOfflineRpcRequestMagic      = 0x51524e4d;
static constexpr uint32_t kOfflineRpcResponseMagic     = 0x53524e4d;
static constexpr uint32_t kOfflineRpcVersion           = 4;
static constexpr uint32_t kOfflineRpcPageBytes         = 4096;
static constexpr uint32_t kOfflineRpcMaxBuffers        = 64;
static constexpr uint32_t kOfflineRpcMaxCommands       = 65536;
static constexpr uint32_t kOfflineRpcBufferOutput      = 1;
static constexpr uint32_t kOfflineRpcVerifyMockMatMul  = 1;
static constexpr uint32_t kOfflineRpcOutputFdIndex     = 1;
static constexpr uint32_t kOfflineRpcOutputOffsetIndex = 2;
static constexpr uint32_t kOfflineRpcOutputSizeIndex   = 3;

struct OfflineRpcRequestHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t bufferCount;
  uint32_t commandCount;
  uint32_t reserved[6];
};

struct OfflineRpcBufferDesc {
  int32_t  id;
  uint32_t logicalSize;
  uint32_t alignment;
  uint32_t flags;
  uint32_t chunkCount;
};

struct OfflineRpcChunkDesc {
  uint32_t offset;
  uint32_t size;
};

struct OfflineRpcCommandDesc {
  uint32_t size;
  uint32_t reserved;
};

struct OfflineRpcResponseHeader {
  uint32_t magic;
  uint32_t version;
  int32_t  status;
  uint32_t outputBytes;
  uint32_t badResults;
  uint32_t firstResult;
  uint64_t averageCycles;
  uint64_t averageTicks;
};

static_assert(sizeof(OfflineRpcRequestHeader) == 40, "Unexpected offline RPC request layout");
static_assert(sizeof(OfflineRpcBufferDesc) == 20, "Unexpected offline RPC buffer layout");
static_assert(sizeof(OfflineRpcChunkDesc) == 8, "Unexpected offline RPC chunk layout");
static_assert(sizeof(OfflineRpcCommandDesc) == 8, "Unexpected offline RPC command layout");
static_assert(sizeof(OfflineRpcResponseHeader) == 40, "Unexpected offline RPC response layout");
