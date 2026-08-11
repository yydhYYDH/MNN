//
//  geometric.cpp
//  MNN
//
//  Created by MNN on 2021/08/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/imgproc/geometric.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace MNN {
namespace CV {

namespace {

constexpr int kPillowResizePrecisionBits = 32 - 8 - 2;
constexpr int kPillowResizePrecision = 1 << kPillowResizePrecisionBits;
constexpr int kOpenCVResizePrecisionBits = 11;
constexpr int kOpenCVResizePrecision = 1 << kOpenCVResizePrecisionBits;

struct ResizeCoefficients {
    int kernelSize = 0;
    std::vector<int> bounds;
    std::vector<int> weights;
};

double pillowCubicFilter(double x) {
    x = std::abs(x);
    if (x < 1.0) {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if (x < 2.0) {
        return (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0;
    }
    return 0.0;
}

int quantizeResizeCoefficient(double value) {
    return static_cast<int>(value < 0.0 ? value - 0.5 : value + 0.5);
}

ResizeCoefficients computePillowResizeCoefficients(int inputSize, int outputSize) {
    constexpr double filterSupport = 2.0;
    const double scale = static_cast<double>(inputSize) / outputSize;
    const double filterScale = std::max(scale, 1.0);
    const double support = filterSupport * filterScale;

    ResizeCoefficients coefficients;
    coefficients.kernelSize = static_cast<int>(std::ceil(support)) * 2 + 1;
    coefficients.bounds.resize(outputSize * 2);
    coefficients.weights.assign(outputSize * coefficients.kernelSize, 0);
    std::vector<double> weights(coefficients.kernelSize);

    for (int output = 0; output < outputSize; ++output) {
        const double center = (output + 0.5) * scale;
        const int first = std::max(static_cast<int>(center - support + 0.5), 0);
        const int count = std::min(static_cast<int>(center + support + 0.5), inputSize) - first;
        double sum = 0.0;
        for (int k = 0; k < count; ++k) {
            weights[k] = pillowCubicFilter((k + first - center + 0.5) / filterScale);
            sum += weights[k];
        }
        if (sum != 0.0) {
            for (int k = 0; k < count; ++k) {
                weights[k] /= sum;
            }
        }
        for (int k = 0; k < count; ++k) {
            coefficients.weights[output * coefficients.kernelSize + k] =
                quantizeResizeCoefficient(weights[k] * kPillowResizePrecision);
        }
        coefficients.bounds[output * 2] = first;
        coefficients.bounds[output * 2 + 1] = count;
    }
    return coefficients;
}

uint8_t clipToByte(int value) {
    return static_cast<uint8_t>(std::max(0, std::min(255, value)));
}

void horizontalPillowCubicResize(const uint8_t* source, int inputWidth, int inputHeight, int channels, uint8_t* dest,
                                 int outputWidth, const ResizeCoefficients& coefficients) {
    constexpr int rounding = 1 << (kPillowResizePrecisionBits - 1);
    for (int y = 0; y < inputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            const int first = coefficients.bounds[x * 2];
            const int count = coefficients.bounds[x * 2 + 1];
            const int* weights = coefficients.weights.data() + x * coefficients.kernelSize;
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int k = 0; k < count; ++k) {
                    sum += source[(y * inputWidth + first + k) * channels + c] * weights[k];
                }
                dest[(y * outputWidth + x) * channels + c] = clipToByte((sum + rounding) >> kPillowResizePrecisionBits);
            }
        }
    }
}

void verticalPillowCubicResize(const uint8_t* source, int width, int inputHeight, int channels, uint8_t* dest,
                               int outputHeight, const ResizeCoefficients& coefficients) {
    constexpr int rounding = 1 << (kPillowResizePrecisionBits - 1);
    for (int y = 0; y < outputHeight; ++y) {
        const int first = coefficients.bounds[y * 2];
        const int count = coefficients.bounds[y * 2 + 1];
        const int* weights = coefficients.weights.data() + y * coefficients.kernelSize;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int k = 0; k < count; ++k) {
                    sum += source[((first + k) * width + x) * channels + c] * weights[k];
                }
                dest[(y * width + x) * channels + c] = clipToByte((sum + rounding) >> kPillowResizePrecisionBits);
            }
        }
    }
}

std::vector<uint8_t> pillowCubicResize8Bit(const uint8_t* source, int inputWidth, int inputHeight, int channels,
                                           int outputWidth, int outputHeight) {
    auto xCoefficients = computePillowResizeCoefficients(inputWidth, outputWidth);
    auto yCoefficients = computePillowResizeCoefficients(inputHeight, outputHeight);
    std::vector<uint8_t> horizontal(static_cast<size_t>(outputWidth) * inputHeight * channels);
    std::vector<uint8_t> output(static_cast<size_t>(outputWidth) * outputHeight * channels);
    horizontalPillowCubicResize(source, inputWidth, inputHeight, channels, horizontal.data(), outputWidth,
                                xCoefficients);
    verticalPillowCubicResize(horizontal.data(), outputWidth, inputHeight, channels, output.data(), outputHeight,
                              yCoefficients);
    return output;
}

void computeOpenCVCubicWeights(float fraction, int* weights) {
    constexpr float cubicParameter = -0.75f;
    const float oneMinusFraction = 1.0f - fraction;
    float coefficients[4];
    coefficients[0] =
        ((cubicParameter * (fraction + 1.0f) - 5.0f * cubicParameter) * (fraction + 1.0f) + 8.0f * cubicParameter) *
            (fraction + 1.0f) -
        4.0f * cubicParameter;
    coefficients[1] = ((cubicParameter + 2.0f) * fraction - (cubicParameter + 3.0f)) * fraction * fraction + 1.0f;
    coefficients[2] =
        ((cubicParameter + 2.0f) * oneMinusFraction - (cubicParameter + 3.0f)) * oneMinusFraction * oneMinusFraction +
        1.0f;
    coefficients[3] = 1.0f - coefficients[0] - coefficients[1] - coefficients[2];
    for (int i = 0; i < 4; ++i) {
        weights[i] = static_cast<int>(std::lrintf(coefficients[i] * kOpenCVResizePrecision));
    }
}

struct OpenCVCubicCoefficients {
    std::vector<int> indices;
    std::vector<int> weights;
};

OpenCVCubicCoefficients computeOpenCVCubicCoefficients(int inputSize, int outputSize) {
    OpenCVCubicCoefficients coefficients;
    coefficients.indices.resize(outputSize * 4);
    coefficients.weights.resize(outputSize * 4);
    const double scale = static_cast<double>(inputSize) / outputSize;
    for (int output = 0; output < outputSize; ++output) {
        const double coordinate = (output + 0.5) * scale - 0.5;
        const int origin = static_cast<int>(std::floor(coordinate));
        computeOpenCVCubicWeights(static_cast<float>(coordinate - origin), coefficients.weights.data() + output * 4);
        for (int k = 0; k < 4; ++k) {
            coefficients.indices[output * 4 + k] = std::max(0, std::min(inputSize - 1, origin + k - 1));
        }
    }
    return coefficients;
}

std::vector<uint8_t> openCVCubicResize8Bit(const uint8_t* source, int inputWidth, int inputHeight, int channels,
                                           int outputWidth, int outputHeight) {
    const auto xCoefficients = computeOpenCVCubicCoefficients(inputWidth, outputWidth);
    const auto yCoefficients = computeOpenCVCubicCoefficients(inputHeight, outputHeight);
    std::vector<int> horizontal(static_cast<size_t>(outputWidth) * inputHeight * channels);
    for (int y = 0; y < inputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int k = 0; k < 4; ++k) {
                    const int inputX = xCoefficients.indices[x * 4 + k];
                    sum += source[(y * inputWidth + inputX) * channels + c] * xCoefficients.weights[x * 4 + k];
                }
                horizontal[(y * outputWidth + x) * channels + c] = sum;
            }
        }
    }

    constexpr int rounding = 1 << (kOpenCVResizePrecisionBits * 2 - 1);
    std::vector<uint8_t> output(static_cast<size_t>(outputWidth) * outputHeight * channels);
    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                int64_t sum = 0;
                for (int k = 0; k < 4; ++k) {
                    const int inputY = yCoefficients.indices[y * 4 + k];
                    sum += static_cast<int64_t>(horizontal[(inputY * outputWidth + x) * channels + c]) *
                           yCoefficients.weights[y * 4 + k];
                }
                output[(y * outputWidth + x) * channels + c] =
                    clipToByte(static_cast<int>((sum + rounding) >> (kOpenCVResizePrecisionBits * 2)));
            }
        }
    }
    return output;
}

} // namespace

std::pair<VARP, VARP> convertMaps(VARP map1, VARP map2, int dstmap1type, bool nninterpolation) {
    // just return src map
    return  { map1, map2 };
}

Matrix getAffineTransform(const Point src[], const Point dst[]) {
    Matrix M;
    M.setPolyToPoly(src, dst, 3);
    return M;
}

Matrix invertAffineTransform(Matrix M) {
    M.invert(&M);
    return M;
}

Matrix getPerspectiveTransform(const Point src[], const Point dst[]) {
    Matrix M;
    M.setPolyToPoly(src, dst, 4);
    return M;
}

VARP getRectSubPix(VARP image, Size patchSize, Point center) {
    // apply below affine:
    // 1, 0, center_x - (width - 1) / 2
    // 0, 1, center_y - (height - 1) / 2
    Matrix M;
    M.setTranslate(center.fX - (patchSize.width - 1) / 2, center.fY - (patchSize.height - 1) / 2);
    return warpAffine(image, M, patchSize, WARP_INVERSE_MAP);
}

Matrix getRotationMatrix2D(Point center, double angle, double scale) {
    Matrix M;
    // rotete with invert equal opencv rotate
    M.setRotate(angle, center.fX, center.fY);
    M.invert(&M);
    // add scale after rotate
    M.postScale(scale, scale, center.fX, center.fY);
    return M;
}

extern std::pair<CV::ImageFormat, CV::ImageFormat> getSrcDstFormat(int code);
extern int format2Channel(CV::ImageFormat format);

VARP remap(VARP src, VARP map1, VARP map2, int interpolation, int borderMode, int borderValue) {
    int oh, ow, oc;
    getVARPSize(map1, &oh, &ow, &oc);
    // src need float, NC4HW4, dims = 4
    auto original_type = src->getInfo()->type;
    src = _Convert(_Unsqueeze(src, {0}), NC4HW4);
    src = _Cast(src, halide_type_of<float>());
    // change remap matrix to gridsmaple matrix: y = (2 * x + 1) / num - 1
    map1 = (map1 * _Scalar<float>(2) + _Scalar<float>(1)) / _Scalar<float>(ow) - _Scalar<float>(1);
    map2 = (map2 * _Scalar<float>(2) + _Scalar<float>(1)) / _Scalar<float>(oh) - _Scalar<float>(1);
    // grid need shape = {n, h, w, 2}
    auto m1info = map1->getInfo();
    auto grid = _Stack({map1, map2}, -1);
    auto ginfo = grid->getInfo();
    grid = _Unsqueeze(grid, {0});
    ginfo = grid->getInfo();
    auto method = InterpolationMethod::BILINEAR;
    if (interpolation == 0) {
        method = InterpolationMethod::NEAREST;
    }
    auto dst = _GridSample(src, grid, method);
    dst = _Squeeze(_Convert(_Cast(dst, original_type), NHWC), {0});
    auto info = dst->getInfo();
    return dst;
}

VARP resize(VARP src, Size dsize, double fx, double fy, int interpolation, int code, std::vector<float> mean, std::vector<float> norm) {
    int ih, iw, ic;
    auto type = src->getInfo()->type;
    getVARPSize(src, &ih, &iw, &ic);
    int oh = dsize.height, ow = dsize.width;
    if (!oh && !ow) {
        oh = ih * fy;
        ow = iw * fx;
    }
    if (ow <= 0 || oh <= 0) {
        MNN_ERROR("Invalid resize output size: %dx%d\n", ow, oh);
        return nullptr;
    }
    fx = static_cast<float>(iw) / ow;
    fy = static_cast<float>(ih) / oh;
    ImageProcess::Config config;
    // cvtColor
    int oc = ic;
    if (code >= 0) {
        auto format = getSrcDstFormat(code);
        config.sourceFormat = format.first;
        config.destFormat = format.second;
        oc = format2Channel(format.second);
    } else {
        ImageFormat format = RGB;
        if (ic == 1) {
            format = GRAY;
        } else if (ic == 4) {
            format = RGBA;
        }
        config.sourceFormat = format;
        config.destFormat = format;
    }
    // toFloat
    auto dstType = type;
    if (!mean.empty() || !norm.empty()) {
        for (int i = 0; i < mean.size() && i < 4; i++) {
            config.mean[i] = mean[i];
        }
        for (int i = 0; i < norm.size() && i < 4; i++) {
            config.normal[i] = norm[i];
        }
        dstType = halide_type_of<float>();
    }
    const uint8_t* source = src->readMap<uint8_t>();
    if (interpolation == INTER_PILLOW_BICUBIC && (type != halide_type_of<uint8_t>() || source == nullptr)) {
        MNN_ERROR("INTER_PILLOW_BICUBIC currently only supports host-mappable uint8 input\n");
        return nullptr;
    }
    std::vector<uint8_t> cubicOutput;
    if ((interpolation == INTER_CUBIC || interpolation == INTER_PILLOW_BICUBIC) && type == halide_type_of<uint8_t>() &&
        source != nullptr && (iw != ow || ih != oh)) {
        cubicOutput = interpolation == INTER_CUBIC ? openCVCubicResize8Bit(source, iw, ih, ic, ow, oh)
                                                   : pillowCubicResize8Bit(source, iw, ih, ic, ow, oh);
        if (code < 0 && mean.empty() && norm.empty()) {
            auto cubicDest = Tensor::create({1, oh, ow, ic}, type);
            ::memcpy(cubicDest->host<uint8_t>(), cubicOutput.data(), cubicOutput.size());
            auto result = Express::Variable::create(Express::Expr::create(cubicDest, true), 0);
            return _Squeeze(result, {0});
        }
        source = cubicOutput.data();
        iw = ow;
        ih = oh;
        fx = 1.0;
        fy = 1.0;
        config.filterType = NEAREST;
    } else if (interpolation == INTER_PILLOW_BICUBIC) {
        // Resizing to the same dimensions is an identity operation. Skip the cubic temporary buffers while still
        // allowing ImageProcess to perform color conversion and normalization.
        config.filterType = NEAREST;
    } else {
        config.filterType = static_cast<Filter>(interpolation);
    }
    std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
    auto dest = Tensor::create({1, oh, ow, oc}, dstType);
    Matrix tr;
    tr.postScale(fx, fy);
    tr.postTranslate(0.5 * (fx - 1), 0.5 * (fy - 1));
    process->setMatrix(tr);
    process->convert(source, iw, ih, 0, dest->host<uint8_t>(), ow, oh, oc, 0, dstType);
    auto res = Express::Variable::create(Express::Expr::create(dest, true), 0);
    return _Squeeze(res, {0});
}

VARP warpAffine(VARP src, Matrix M, Size dsize, int flags, int borderMode, int borderValue, int code, std::vector<float> mean, std::vector<float> norm) {
    int ih, iw, ic;
    auto type = src->getInfo()->type;
    getVARPSize(src, &ih, &iw, &ic);
    int oh = dsize.height, ow = dsize.width;
    // auto dest = Tensor::create({1, oh, ow, ic}, type);
    ImageProcess::Config config;
    config.filterType = flags < 3 ? static_cast<Filter>(flags) : BILINEAR;
    switch (borderMode) {
        case BORDER_CONSTANT:
            config.wrap = ZERO;
            break;
        case BORDER_REPLICATE:
            config.wrap = REPEAT;
            break;
        case BORDER_TRANSPARENT:
            config.wrap = CLAMP_TO_EDGE;
            break;
        default:
            MNN_ERROR("Don't support borderMode!");
            break;
    }
    // cvtColor
    int oc = ic;
    if (code >= 0) {
        auto format = getSrcDstFormat(code);
        config.sourceFormat = format.first;
        config.destFormat = format.second;
        oc = format2Channel(format.second);
    } else {
        ImageFormat format = RGB;
        if (ic == 1) {
            format = GRAY;
        } else if (ic == 4) {
            format = RGBA;
        }
        config.sourceFormat = format;
        config.destFormat = format;
    }
    // toFloat
    auto dstType = type;
    if (!mean.empty() || !norm.empty()) {
        for (int i = 0; i < mean.size() && i < 4; i++) {
            config.mean[i] = mean[i];
        }
        for (int i = 0; i < norm.size() && i < 4; i++) {
            config.normal[i] = norm[i];
        }
        dstType = halide_type_of<float>();
    }
    auto dest = Tensor::create({1, oh, ow, oc}, dstType);
    std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
    if (flags != WARP_INVERSE_MAP) {
        bool invert = M.invert(&M);
        MNN_ASSERT(invert);
    }
    process->setMatrix(M);
    process->setPadding(borderValue);
    process->convert(src->readMap<uint8_t>(), iw, ih, 0, dest->host<uint8_t>(), ow, oh, oc, 0, dstType);
    auto res = Express::Variable::create(Express::Expr::create(dest, true), 0);
    return _Squeeze(res, {0});
}

VARP warpPerspective(VARP src, Matrix M, Size dsize, int flags, int borderMode, int borderValue) {
    return warpAffine(src, M, dsize, flags, borderMode, borderValue);
}

VARP undistortPoints(VARP src, VARP cameraMatrix, VARP distCoeffs) {
    // Don't support distCoeffs
    auto dims = src->getInfo()->dim;
    int n = dims[0];
    auto dst  = _Input(dims, NCHW);
    auto iptr = src->readMap<float>();
    auto optr = dst->writeMap<float>();
    auto cptr = cameraMatrix->readMap<float>();
    double fx = cptr[0];
    double fy = cptr[4];
    double ifx = 1./fx;
    double ify = 1./fy;
    double cx = cptr[2];
    double cy = cptr[5];
    for (int i = 0; i < n; i++) {
        auto x = iptr[i * 2], y = iptr[i * 2 + 1];
        auto u = x;
        auto v = y;
        x = (x - cx)*ifx;
        y = (y - cy)*ify;
        optr[i * 2] = x;
        optr[i * 2 + 1] = y;
    }
    return dst;
}

} // CV
} // namespace MNN
