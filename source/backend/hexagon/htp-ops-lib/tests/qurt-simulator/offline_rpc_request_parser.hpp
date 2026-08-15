#pragma once

#include "offline_rpc_protocol.h"

#include <cstdint>
#include <vector>

struct OfflineRpcRequest {
    OfflineRpcRequestHeader header = {};
    std::vector<OfflineRpcBufferDesc> buffers;
    std::vector<OfflineRpcCommandDesc> commands;
    std::vector<std::vector<uint8_t>> commandData;
    std::vector<std::vector<OfflineRpcChunkDesc>> chunks;
};

bool readOfflineRpcRequest(const char* path, OfflineRpcRequest* request, bool loadCommandData = true);
