#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <set>
#include <string>
#include <vector>

#include "schema/current/MNN_generated.h"

using namespace MNN::Express;

namespace {

void printVariable(const char *prefix, size_t index, const VARP &variable) {
  const auto *info = variable->getInfo();
  printf("%s[%zu]: name=%s", prefix, index, variable->name().c_str());
  if (info == nullptr) {
    printf(" info=unknown\n");
    return;
  }
  printf(" elements=%zu type=(code=%d bits=%d) order=%d dims=", info->size, info->type.code, info->type.bits,
         info->order);
  for (size_t dim = 0; dim < info->dim.size(); ++dim) {
    printf("%s%d", dim == 0 ? "" : "x", info->dim[dim]);
  }
  printf("\n");
}

int listVariables(const char *modelPath) {
  const auto variables = Variable::loadMap(modelPath);
  if (variables.empty()) {
    fprintf(stderr, "No variables found in %s\n", modelPath);
    return 2;
  }
  size_t index = 0;
  for (const auto &item : variables) {
    printVariable("variable", index++, item.second);
  }
  const auto io = Variable::getInputAndOutput(variables);
  index         = 0;
  for (const auto &item : io.first) {
    printVariable("input", index++, item.second);
  }
  index = 0;
  for (const auto &item : io.second) {
    printVariable("output", index++, item.second);
  }
  return 0;
}

int listConvolutions(const char *modelPath) {
  const auto variables = Variable::loadMap(modelPath);
  size_t     index     = 0;
  for (const auto &item : variables) {
    const auto  expr = item.second->expr().first;
    const auto *op   = expr != nullptr ? expr->get() : nullptr;
    if (op == nullptr || (op->type() != MNN::OpType_Convolution && op->type() != MNN::OpType_ConvolutionDepthwise)) {
      continue;
    }
    const auto *conv   = op->main_as_Convolution2D();
    const auto *common = conv != nullptr ? conv->common() : nullptr;
    printf("conv[%zu]: output=%s expr=%s inputs=%zu", index++, item.first.c_str(), expr->name().c_str(),
           expr->inputs().size());
    if (common != nullptr) {
      printf(" ic=%d oc=%d kernel=%dx%d group=%d", common->inputCount(), common->outputCount(), common->kernelX(),
             common->kernelY(), common->group());
    }
    printf(" weight=%u bias=%u\n", conv != nullptr && conv->weight() != nullptr ? conv->weight()->size() : 0,
           conv != nullptr && conv->bias() != nullptr ? conv->bias()->size() : 0);
  }
  return index == 0 ? 3 : 0;
}

int extractConvolution(const char *modelPath, const char *outputName, const char *outputPath, int rows) {
  const auto variables = Variable::loadMap(modelPath);
  const auto iter      = variables.find(outputName);
  if (iter == variables.end()) {
    fprintf(stderr, "Convolution output not found: %s\n", outputName);
    return 2;
  }
  const auto  sourceExpr = iter->second->expr().first;
  const auto *op         = sourceExpr != nullptr ? sourceExpr->get() : nullptr;
  const auto *conv       = op != nullptr ? op->main_as_Convolution2D() : nullptr;
  const auto *common     = conv != nullptr ? conv->common() : nullptr;
  if (op == nullptr || common == nullptr || op->type() != MNN::OpType_Convolution || rows <= 0) {
    fprintf(stderr, "Variable is not an extractable convolution: %s\n", outputName);
    return 3;
  }
  auto input = _Input({ 1, common->inputCount(), 1, rows }, NCHW, halide_type_of<float>());
  input->setName("input");
  std::vector<VARP> opInputs{ input };
  VARP output = Variable::create(Expr::create(sourceExpr->extra(), std::move(opInputs), sourceExpr->outputSize()));
  output      = _Convert(output, NCHW);
  output->setName("output");
  Variable::save(std::vector<VARP>{ output }, outputPath);
  printf("extracted: source=%s output=%s rows=%d ic=%d oc=%d model=%s\n", modelPath, outputName, rows,
         common->inputCount(), common->outputCount(), outputPath);
  return 0;
}

void traceExpression(const VARP &variable, int depth, int maxDepth, std::set<const Expr *> *visited) {
  const auto expr = variable->expr().first;
  if (expr == nullptr || !visited->insert(expr.get()).second) {
    return;
  }
  const auto *op = expr->get();
  printf("%*s%s <- %s (%s) inputs=%zu\n", depth * 2, "", variable->name().c_str(), expr->name().c_str(),
         op != nullptr ? MNN::EnumNameOpType(op->type()) : "InputOrConst", expr->inputs().size());
  if (depth >= maxDepth) {
    return;
  }
  for (const auto &input : expr->inputs()) {
    traceExpression(input, depth + 1, maxDepth, visited);
  }
}

int traceVariable(const char *modelPath, const char *outputName, int maxDepth) {
  const auto variables = Variable::loadMap(modelPath);
  const auto iter      = variables.find(outputName);
  if (iter == variables.end()) {
    fprintf(stderr, "Variable not found: %s\n", outputName);
    return 2;
  }
  std::set<const Expr *> visited;
  traceExpression(iter->second, 0, maxDepth, &visited);
  return 0;
}

int extractVisionBlock(const char *modelPath, const char *inputName, const char *outputName, const char *outputPath,
                       int rows) {
  auto       variables    = Variable::loadMap(modelPath);
  const auto inputIter    = variables.find(inputName);
  const auto outputIter   = variables.find(outputName);
  const auto positionIter = variables.find("position_ids");
  const auto maskIter     = variables.find("attention_mask");
  if (inputIter == variables.end() || outputIter == variables.end() || positionIter == variables.end() ||
      maskIter == variables.end() || rows <= 0) {
    fprintf(stderr, "Unable to locate Vision block boundaries or dynamic inputs\n");
    return 2;
  }
  auto input         = _Input({ 1, rows, 1024 }, NCHW, halide_type_of<float>());
  auto positionIds   = _Input({ 2, rows }, NCHW, halide_type_of<int32_t>());
  auto attentionMask = _Input({ 1, rows, rows }, NCHW, halide_type_of<float>());
  input->setName("input");
  positionIds->setName("position_ids");
  attentionMask->setName("attention_mask");
  Variable::replace(inputIter->second, input);
  Variable::replace(positionIter->second, positionIds);
  Variable::replace(maskIter->second, attentionMask);
  auto output = outputIter->second;
  output->setName("output");
  Variable::save(std::vector<VARP>{ output }, outputPath);
  printf("extracted-block: input=%s output=%s rows=%d model=%s\n", inputName, outputName, rows, outputPath);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 3 && std::string(argv[1]) == "list") {
    return listVariables(argv[2]);
  }
  if (argc == 3 && std::string(argv[1]) == "convs") {
    return listConvolutions(argv[2]);
  }
  if (argc == 6 && std::string(argv[1]) == "extract-conv") {
    return extractConvolution(argv[2], argv[3], argv[4], atoi(argv[5]));
  }
  if (argc == 5 && std::string(argv[1]) == "trace") {
    return traceVariable(argv[2], argv[3], atoi(argv[4]));
  }
  if (argc == 7 && std::string(argv[1]) == "extract-block") {
    return extractVisionBlock(argv[2], argv[3], argv[4], argv[5], atoi(argv[6]));
  }
  if (argc != 3) {
    fprintf(stderr,
            "Usage: %s list MODEL.mnn | convs MODEL.mnn | trace MODEL.mnn OUTPUT_NAME DEPTH | extract-conv "
            "MODEL.mnn OUTPUT_NAME OUTPUT.mnn ROWS | extract-block MODEL.mnn INPUT_NAME OUTPUT_NAME OUTPUT.mnn "
            "ROWS | INPUT.mnn OUTPUT_PREFIX\n",
            argv[0]);
    return 64;
  }
  auto variables = Variable::load(argv[1]);
  if (variables.empty()) {
    fprintf(stderr, "No variables found in %s\n", argv[1]);
    return 2;
  }
  for (size_t index = 0; index < variables.size(); ++index) {
    auto variable = variables[index];
    auto info     = variable->getInfo();
    if (info == nullptr) {
      return 3;
    }
    if (info->order == NC4HW4 && info->dim.size() > 1) {
      variable = _Convert(variable, NCHW);
      info     = variable->getInfo();
    }
    if (info->type.code != halide_type_float) {
      variable = _Cast<float>(variable);
      info     = variable->getInfo();
    }
    const float *data = variable->readMap<float>();
    if (data == nullptr) {
      return 4;
    }
    const std::string path = std::string(argv[2]) + "." + std::to_string(index) + ".f32";
    std::ofstream     output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(data), info->size * sizeof(float));
    if (!output) {
      return 5;
    }
    size_t nonFinite = 0;
    for (size_t element = 0; element < info->size; ++element) {
      if (!std::isfinite(data[element])) {
        ++nonFinite;
      }
    }
    printf("variable[%zu]: name=%s elements=%zu bytes=%zu non_finite=%zu dims=", index, variable->name().c_str(),
           info->size, info->size * sizeof(float), nonFinite);
    for (size_t dim = 0; dim < info->dim.size(); ++dim) {
      printf("%s%d", dim == 0 ? "" : "x", info->dim[dim]);
    }
    printf(" file=%s\n", path.c_str());
  }
  return 0;
}
