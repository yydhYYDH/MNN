//
//  geometric_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "test_env.hpp"
#include "cv/imgcodecs.hpp"

#ifdef MNN_GEOMETRIC_TEST

static Env<uint8_t> testEnv(img_name, false);

// convertMaps
TEST(convertMaps, basic) {
    const int h = testEnv.cvSrc.rows;
    const int w = testEnv.cvSrc.cols;
    cv::Mat mapx, mapy, map_x, map_y;
    mapx.create(testEnv.cvSrc.size(), CV_32FC1);
    mapy.create(testEnv.cvSrc.size(), CV_32FC1);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            mapx.at<float>(j, i) = w - i;
            mapy.at<float>(j, i) = h - j;
        }
    }
    VARP mapX = _Const(mapx.ptr(), {h, w}, NHWC, halide_type_of<float>());
    VARP mapY = _Const(mapy.ptr(), {h, w}, NHWC, halide_type_of<float>());
    cv::convertMaps(mapx, mapy, map_x, map_y, CV_16SC2);
    cv::remap(testEnv.cvSrc, testEnv.cvDst, map_x, map_y, INTER_LINEAR);
    auto mapXY = convertMaps(mapX, mapY, CV_16SC2);
    testEnv.mnnDst = remap(testEnv.mnnSrc, mapXY.first, mapXY.second, INTER_LINEAR);
    EXPECT_TRUE(testEnv.equal());
}

// getAffineTransform
TEST(getAffineTransform, basic) {
    float points[] = { 50, 50, 200, 50, 50, 200, 10, 100, 200, 20, 100, 250 };
    cv::Point2f cvSrc[3], cvDst[3];
    memcpy(cvSrc, points, 6 * sizeof(float));
    memcpy(cvDst, points + 6, 6 * sizeof(float));
    Point mnnSrc[3], mnnDst[3];
    memcpy(mnnSrc, points, 6 * sizeof(float));
    memcpy(mnnDst, points + 6, 6 * sizeof(float));
    cv::Mat cvTrans_double = cv::getAffineTransform(cvSrc, cvDst);
    cv::Mat cvTrans;
    cvTrans_double.convertTo(cvTrans, CV_32F);
    Matrix mnnTrans = getAffineTransform(mnnSrc, mnnDst);
    EXPECT_TRUE(testEnv.equal(cvTrans, mnnTrans));
}

// getPerspectiveTransform
TEST(getPerspectiveTransform, basic_1) {
    float points[] = { 0, 0, 50, 50, 200, 50, 50, 200, 5, 5, 10, 100, 200, 20, 100, 250 };
    cv::Point2f cvSrc[4], cvDst[4];
    memcpy(cvSrc, points, 8 * sizeof(float));
    memcpy(cvDst, points + 8, 8 * sizeof(float));
    Point mnnSrc[4], mnnDst[4];
    memcpy(mnnSrc, points, 8 * sizeof(float));
    memcpy(mnnDst, points + 8, 8 * sizeof(float));
    cv::Mat cvTrans_double = cv::getPerspectiveTransform(cvSrc, cvDst);
    cv::Mat cvTrans;
    cvTrans_double.convertTo(cvTrans, CV_32F);
    Matrix mnnTrans = getPerspectiveTransform(mnnSrc, mnnDst);
    EXPECT_TRUE(testEnv.equal(cvTrans, mnnTrans));
}

TEST(getPerspectiveTransform, basic_2) {
    float points[] = { 0, 0, 479, 0, 0, 359, 479, 359, 0, 46.8, 432, 0, 96, 252, 384, 360 };
    cv::Point2f cvSrc[4], cvDst[4];
    memcpy(cvSrc, points, 8 * sizeof(float));
    memcpy(cvDst, points + 8, 8 * sizeof(float));
    Point mnnSrc[4], mnnDst[4];
    memcpy(mnnSrc, points, 8 * sizeof(float));
    memcpy(mnnDst, points + 8, 8 * sizeof(float));
    cv::Mat cvTrans_double = cv::getPerspectiveTransform(cvSrc, cvDst);
    cv::Mat cvTrans;
    cvTrans_double.convertTo(cvTrans, CV_32F);
    std::cout << cvTrans;
    Matrix mnnTrans = getPerspectiveTransform(mnnSrc, mnnDst);
    EXPECT_TRUE(testEnv.equal(cvTrans, mnnTrans));
}

// getRotationMatrix2D
TEST(getRotationMatrix2D, basic) {
    cv::Point2f cvCenter {10, 10};
    Point mnnCenter {10, 10};
    cv::Mat cvTrans_double = cv::getRotationMatrix2D(cvCenter, 50, 0.6);
    cv::Mat cvTrans;
    cvTrans_double.convertTo(cvTrans, CV_32F);
    Matrix mnnTrans = getRotationMatrix2D(mnnCenter, 50, 0.6);
    EXPECT_TRUE(testEnv.equal(cvTrans, mnnTrans));
}

// getRectSubPix
TEST(getRectSubPix, basic) {
    cv::Point2f cvCenter {10, 10};
    Point mnnCenter {10, 10};
    cv::getRectSubPix(testEnv.cvSrc, {11, 11}, cvCenter, testEnv.cvDst);
    testEnv.mnnDst = getRectSubPix(testEnv.mnnSrc, {11, 11}, mnnCenter);
    EXPECT_TRUE(testEnv.equal());
}

// invertAffineTransform
TEST(invertAffineTransform, basic) {
    std::vector<float> M { 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1), cvDst = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM, mnnDst;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::invertAffineTransform(cvM, cvDst);
    mnnDst = invertAffineTransform(mnnM);
    EXPECT_TRUE(testEnv.equal(cvDst, mnnDst));
}

// remap
TEST(remap, rotate) {
    const int h = testEnv.cvSrc.rows;
    const int w = testEnv.cvSrc.cols;
    cv::Mat mapx, mapy;
    mapx.create(testEnv.cvSrc.size(), CV_32FC1);
    mapy.create(testEnv.cvSrc.size(), CV_32FC1);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            mapx.at<float>(j, i) = w - i;
            mapy.at<float>(j, i) = h - j;
        }
    }
    VARP mapX = _Const(mapx.ptr(), {h, w}, NHWC, halide_type_of<float>());
    VARP mapY = _Const(mapy.ptr(), {h, w}, NHWC, halide_type_of<float>());
    cv::remap(testEnv.cvSrc, testEnv.cvDst, mapx, mapy, INTER_LINEAR);
    testEnv.mnnDst = remap(testEnv.mnnSrc, mapX, mapY, INTER_LINEAR);
    EXPECT_TRUE(testEnv.equal());
}

TEST(remap, scale) {
    const int h = testEnv.cvSrc.rows;
    const int w = testEnv.cvSrc.cols;
    cv::Mat mapx, mapy;
    mapx.create(testEnv.cvSrc.size(), CV_32FC1);
    mapy.create(testEnv.cvSrc.size(), CV_32FC1);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            if (i > w * 0.25 && i < w * 0.75 && j > h * 0.25 && j < h * 0.75) {
                mapx.at<float>(j, i) = 2 * (i - w * 0.25) + 0.5;
                mapy.at<float>(j, i) = 2 * (j - h * 0.25) + 0.5;
            } else {
                mapx.at<float>(j, i) = 0;
                mapy.at<float>(j, i) = 0;
            }
        }
    }
    VARP mapX = _Const(mapx.ptr(), {h, w}, NHWC, halide_type_of<float>());
    VARP mapY = _Const(mapy.ptr(), {h, w}, NHWC, halide_type_of<float>());
    cv::remap(testEnv.cvSrc, testEnv.cvDst, mapx, mapy, INTER_LINEAR);
    testEnv.mnnDst = remap(testEnv.mnnSrc, mapX, mapY, INTER_LINEAR);
    EXPECT_TRUE(testEnv.equal());
}
// resize
TEST(resize, x3_x0_5) {
    cv::resize(testEnv.cvSrc, testEnv.cvDst, cv::Size(), 3, 0.5);
    testEnv.mnnDst = resize(testEnv.mnnSrc, {}, 3, 0.5);
    EXPECT_TRUE(testEnv.equal());
}

TEST(resize, x2_x2) {
    cv::resize(testEnv.cvSrc, testEnv.cvDst, cv::Size(), 2, 2);
    testEnv.mnnDst = resize(testEnv.mnnSrc, {}, 2, 2);
    EXPECT_TRUE(testEnv.equal());
}

TEST(resize, 100_100) {
    cv::resize(testEnv.cvSrc, testEnv.cvDst, cv::Size(200, 200));
    testEnv.mnnDst = resize(testEnv.mnnSrc, {200, 200});
    EXPECT_TRUE(testEnv.equal());
}

TEST(resize, 960_720) {
    cv::resize(testEnv.cvSrc, testEnv.cvDst, cv::Size(960, 720));
    testEnv.mnnDst = resize(testEnv.mnnSrc, {960, 720});
    EXPECT_TRUE(testEnv.equal());
}

TEST(resize, pillow_bicubic_8bit) {
    const int channels = 3;
    auto check = [channels](int inputWidth, int inputHeight, int outputWidth, int outputHeight,
                            const std::vector<uint8_t>& expected) {
        std::vector<uint8_t> input(static_cast<size_t>(inputWidth) * inputHeight * channels);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<uint8_t>((i * 37 + (i / channels) * 13 + 17) % 256);
        }
        auto source = _Const(input.data(), {inputHeight, inputWidth, channels}, NHWC, halide_type_of<uint8_t>());
        auto output = resize(source, {outputWidth, outputHeight}, 0, 0, INTER_PILLOW_BICUBIC);
        auto outputPtr = output->readMap<uint8_t>();
        ASSERT_NE(outputPtr, nullptr);
        EXPECT_EQ(0, ::memcmp(outputPtr, expected.data(), expected.size()));
    };

    check(3, 2, 5, 4,
          {0,   33,  70,  65,  102, 139, 153, 190, 227, 60,  97,  134, 0,   24, 61,  36,  73,  110, 69,  106,
           143, 112, 149, 186, 64,  101, 138, 28,  64,  101, 115, 152, 189, 78, 115, 152, 30,  67,  104, 73,
           110, 147, 106, 143, 180, 155, 192, 229, 82,  119, 156, 0,   26,  63, 77,  114, 151, 146, 183, 220});
    check(7, 5, 3, 2, {133, 123, 123, 126, 112, 119, 149, 105, 108, 123, 157, 113, 113, 146, 148, 102, 139, 142});
}

TEST(resize, opencv_cubic_8bit) {
    const int channels = 3;
    auto check = [channels](int inputWidth, int inputHeight, int outputWidth, int outputHeight,
                            const std::vector<uint8_t>& expected) {
        std::vector<uint8_t> input(static_cast<size_t>(inputWidth) * inputHeight * channels);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<uint8_t>((i * 37 + (i / channels) * 13 + 17) % 256);
        }
        auto source = _Const(input.data(), {inputHeight, inputWidth, channels}, NHWC, halide_type_of<uint8_t>());
        auto output = resize(source, {outputWidth, outputHeight}, 0, 0, INTER_CUBIC);
        auto outputPtr = output->readMap<uint8_t>();
        ASSERT_NE(outputPtr, nullptr);
        EXPECT_EQ(0, ::memcmp(outputPtr, expected.data(), expected.size()));
    };

    // Generated with OpenCV's scalar INTER_CUBIC path (cubic parameter -0.75, 11-bit fixed-point coefficients).
    check(3, 2, 5, 4,
          {0,   27,  64,  75,  112, 149, 156, 193, 230, 69,  106, 143, 0,   18,  55,  37,  74,  111, 74,  111,
           148, 109, 146, 183, 69,  106, 143, 28,  65,  102, 114, 151, 188, 73,  110, 147, 33,  70,  107, 68,
           105, 142, 179, 161, 198, 235, 73,  110, 147, 0,   23,  60,  67,  104, 141, 152, 189, 226});
    check(7, 5, 3, 2, {205, 63, 105, 227, 39, 76, 204, 43, 86, 142, 179, 42, 152, 189, 226, 123, 160, 202});
}

// warpAffine
TEST(warpAffine, scale) {
    std::vector<float> M { 0.5, 0, 0, 0, 0.8, 0 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360});
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360});
    EXPECT_TRUE(testEnv.equal());
}

TEST(warpAffine, scale_trans) {
    std::vector<float> M { 0.5, 0, 1, 0, 0.8, 2 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360});
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360});
    EXPECT_TRUE(testEnv.equal());
}

TEST(warpAffine, skew) {
    std::vector<float> M { 0, 0.5, 0, 0.5, 0, 0 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360});
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360});
    EXPECT_TRUE(testEnv.equal());
}

TEST(warpAffine, trans_1_1_default) {
    std::vector<float> M { 1, 0, 1, 0, 1, 1 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360});
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360});
    EXPECT_TRUE(testEnv.equal());
}

TEST(warpAffine, trans_2_2_inverse) {
    std::vector<float> M { 1, 0, 2, 0, 1, 2 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360}, cv::WARP_INVERSE_MAP);
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360}, WARP_INVERSE_MAP);
    EXPECT_TRUE(testEnv.equal());
}

TEST(warpAffine, trans_3_3_replicate) {
    std::vector<float> M { 1, 0, 3, 0, 1, 3 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360}, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360}, INTER_LINEAR, BORDER_REPLICATE);
    EXPECT_TRUE(testEnv.equal());
}

TEST(warpAffine, trans_3_3_transparent) {
    std::vector<float> M { 1, 0, 3, 0, 1, 3 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360}, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360}, INTER_LINEAR, BORDER_TRANSPARENT);
    EXPECT_TRUE(testEnv.equal());
}

TEST(warpAffine, trans_3_3_constant_5) {
    std::vector<float> M { 1, 0, 3, 0, 1, 3 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);;
    cv::warpAffine(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360}, cv::INTER_LINEAR, cv::BORDER_CONSTANT, {5, 5, 5, 5});
    testEnv.mnnDst = warpAffine(testEnv.mnnSrc, mnnM, {480, 360}, INTER_LINEAR, BORDER_CONSTANT, 5);
    EXPECT_TRUE(testEnv.equal());
}

#if 0
// warpPerspective
TEST(warpPerspective, trans_1_1_default) {
    std::vector<float> M { 0.40369818, 0.37649557, 0,
                           -0.097703546, 0.85793871, 46.799999,
                           -0.0011531961, 0.0011363134, 1 };
    cv::Mat cvM = cv::Mat(3, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    cv::warpPerspective(testEnv.cvSrc, testEnv.cvDst, cvM, {480, 360});
    testEnv.mnnDst = warpPerspective(testEnv.mnnSrc, mnnM, {480, 360});
    EXPECT_TRUE(testEnv.equal());
}
#endif

#endif
