#include "tokenizer/tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Transformer;

static std::string readText(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::binary);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " tokenizer.mtok text_file\n";
        return 1;
    }
    std::unique_ptr<Tokenizer> tokenizer(Tokenizer::createTokenizer(argv[1]));
    if (!tokenizer) {
        std::cerr << "failed to load tokenizer: " << argv[1] << "\n";
        return 2;
    }
    auto text = readText(argv[2]);
    auto ids = tokenizer->encode(text);
    std::cout << "count=" << ids.size() << "\n";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) {
            std::cout << ' ';
        }
        std::cout << ids[i];
    }
    std::cout << "\n";
    return 0;
}
