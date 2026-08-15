#include "offline_rpc_request_parser.hpp"

#include <fstream>
#include <limits>

namespace {

bool readBytes(std::ifstream* input, void* data, uint64_t size, uint64_t* offset, uint64_t fileSize) {
    if (*offset > fileSize || size > fileSize - *offset || size > std::numeric_limits<size_t>::max()) {
        return false;
    }
    input->read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!*input) {
        return false;
    }
    *offset += size;
    return true;
}

bool skipBytes(std::ifstream* input, uint64_t size, uint64_t* offset, uint64_t fileSize) {
    if (*offset > fileSize || size > fileSize - *offset ||
        size > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }
    input->seekg(static_cast<std::streamoff>(size), std::ios::cur);
    if (!*input) {
        return false;
    }
    *offset += size;
    return true;
}

}  // namespace

bool readOfflineRpcRequest(const char* path, OfflineRpcRequest* request, bool loadCommandData) {
    if (request == nullptr) {
        return false;
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::streamoff end = input.tellg();
    if (!input || end < static_cast<std::streamoff>(sizeof(OfflineRpcRequestHeader))) {
        return false;
    }
    const uint64_t fileSize = static_cast<uint64_t>(end);
    uint64_t offset = 0;
    input.seekg(0);
    if (!readBytes(&input, &request->header, sizeof(request->header), &offset, fileSize) ||
        request->header.magic != kOfflineRpcRequestMagic || request->header.version != kOfflineRpcVersion ||
        request->header.bufferCount == 0 || request->header.bufferCount > kOfflineRpcMaxBuffers ||
        request->header.commandCount == 0 || request->header.commandCount > kOfflineRpcMaxCommands) {
        return false;
    }

    request->buffers.resize(request->header.bufferCount);
    request->commands.resize(request->header.commandCount);
    request->commandData.resize(request->header.commandCount);
    request->chunks.resize(request->header.bufferCount);
    if (!readBytes(&input, request->buffers.data(), request->buffers.size() * sizeof(request->buffers[0]), &offset,
                   fileSize) ||
        !readBytes(&input, request->commands.data(), request->commands.size() * sizeof(request->commands[0]), &offset,
                   fileSize)) {
        return false;
    }
    for (size_t i = 0; i < request->commands.size(); ++i) {
        const uint32_t size = request->commands[i].size;
        if (size == 0 || size > 65536) {
            return false;
        }
        if (loadCommandData) {
            request->commandData[i].resize(size);
            if (!readBytes(&input, request->commandData[i].data(), size, &offset, fileSize)) {
                return false;
            }
        } else if (!skipBytes(&input, size, &offset, fileSize)) {
            return false;
        }
    }
    for (size_t i = 0; i < request->buffers.size(); ++i) {
        const auto& buffer = request->buffers[i];
        const uint64_t remainingBytes = fileSize - offset;
        if (buffer.chunkCount > remainingBytes / sizeof(OfflineRpcChunkDesc)) {
            return false;
        }
        for (uint32_t j = 0; j < buffer.chunkCount; ++j) {
            OfflineRpcChunkDesc chunk = {};
            if (!readBytes(&input, &chunk, sizeof(chunk), &offset, fileSize) ||
                chunk.size == 0 || chunk.offset > buffer.logicalSize ||
                chunk.size > buffer.logicalSize - chunk.offset ||
                !skipBytes(&input, chunk.size, &offset, fileSize)) {
                return false;
            }
            request->chunks[i].emplace_back(chunk);
        }
    }
    return offset == fileSize;
}
