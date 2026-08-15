#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "schema/current/MNN_generated.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

using namespace MNN;
using namespace MNN::Express;

namespace {

void writeInput(const char* path, size_t count, int seed) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (size_t i = 0; i < count; ++i) {
        const float value = static_cast<float>((static_cast<int>(i) * 13 + seed * 7) % 29 - 14) / 32.0f;
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

int generateFlashAttention(const char* modelPath, const char* manifestPath, int seqLen, int heads, int kvHeads,
                           int headDim) {
    auto query = _Input({1, seqLen, heads, headDim}, NCHW, halide_type_of<float>());
    auto key = _Input({1, seqLen, kvHeads, headDim}, NCHW, halide_type_of<float>());
    auto value = _Input({seqLen, kvHeads * headDim}, NC4HW4, halide_type_of<float>());
    query->setName("query");
    key->setName("key");
    value->setName("value");

    std::shared_ptr<OpT> op(new OpT);
    op->type = OpType_Attention;
    op->main.type = OpParameter_AttentionParam;
    op->main.value = new AttentionParamT;
    op->main.AsAttentionParam()->kv_cache = false;
    op->main.AsAttentionParam()->output_c4 = true;
    op->main.AsAttentionParam()->attnScale = 1.0f / std::sqrt(static_cast<float>(headDim));
    auto output = Variable::create(Expr::create(op.get(), {query, key, value}));
    output->setName("output");
    Variable::save({output}, modelPath);

    std::ofstream manifest(manifestPath, std::ios::trunc);
    const std::string prefix = std::string(manifestPath) + ".";
    const size_t qSize = static_cast<size_t>(seqLen) * heads * headDim;
    const size_t kvSize = static_cast<size_t>(seqLen) * kvHeads * headDim;
    writeInput((prefix + "query").c_str(), qSize, 1);
    writeInput((prefix + "key").c_str(), kvSize, 2);
    writeInput((prefix + "value").c_str(), kvSize, 3);
    manifest << "query " << prefix << "query\n";
    manifest << "key " << prefix << "key\n";
    manifest << "value " << prefix << "value\n";
    return manifest ? 0 : 2;
}

int generateSoftmax(const char* modelPath, const char* manifestPath) {
    auto input = _Input({1, 4, 16, 1}, NCHW, halide_type_of<float>());
    input->setName("input");
    auto output = _Softmax(input, 2);
    output->setName("output");
    Variable::save({output}, modelPath);

    const std::string prefix = std::string(manifestPath) + ".";
    writeInput((prefix + "input").c_str(), 64, 4);
    std::ofstream manifest(manifestPath, std::ios::trunc);
    manifest << "input " << prefix << "input\n";
    return manifest ? 0 : 2;
}

int generateAdd(const char* modelPath, const char* manifestPath) {
    auto lhs = _Input({1, 4, 16, 1}, NCHW, halide_type_of<float>());
    auto rhs = _Input({1, 4, 16, 1}, NCHW, halide_type_of<float>());
    lhs->setName("lhs");
    rhs->setName("rhs");
    auto output = _Add(lhs, rhs);
    output->setName("output");
    Variable::save({output}, modelPath);

    const std::string prefix = std::string(manifestPath) + ".";
    writeInput((prefix + "lhs").c_str(), 64, 5);
    writeInput((prefix + "rhs").c_str(), 64, 6);
    std::ofstream manifest(manifestPath, std::ios::trunc);
    manifest << "lhs " << prefix << "lhs\n";
    manifest << "rhs " << prefix << "rhs\n";
    return manifest ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 8) {
        fprintf(stderr, "Usage: %s flash-attention MODEL MANIFEST [SEQ HEADS KV_HEADS HEAD_DIM] | softmax MODEL MANIFEST | add MODEL MANIFEST\n",
                argv[0]);
        return 64;
    }
    if (strcmp(argv[1], "softmax") == 0 && argc == 4) {
        return generateSoftmax(argv[2], argv[3]);
    }
    if (strcmp(argv[1], "add") == 0 && argc == 4) {
        return generateAdd(argv[2], argv[3]);
    }
    if (strcmp(argv[1], "flash-attention") != 0) {
        return 64;
    }
    const int seqLen = argc == 8 ? atoi(argv[4]) : 1;
    const int heads = argc == 8 ? atoi(argv[5]) : 4;
    const int kvHeads = argc == 8 ? atoi(argv[6]) : 2;
    const int headDim = argc == 8 ? atoi(argv[7]) : 64;
    if (seqLen <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads % kvHeads != 0) {
        return 65;
    }
    return generateFlashAttention(argv[2], argv[3], seqLen, heads, kvHeads, headDim);
}
