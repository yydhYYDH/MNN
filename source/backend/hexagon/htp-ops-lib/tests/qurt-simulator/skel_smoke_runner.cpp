#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <remote.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
AEEResult htp_ops_open(const char* uri, remote_handle64* handle);
AEEResult htp_ops_close(remote_handle64 handle);
AEEResult htp_ops_init_backend(remote_handle64 handle);
AEEResult htp_ops_get_skel_arch(remote_handle64 handle, uint32* arch);
int vtcm_manager_acquire(void);
void vtcm_manager_release(void);
int hmx_matmulq4fp16_mle32(uint8_t* output, const uint8_t* activation, const uint8_t* weight,
                           const uint8_t* scale, const uint8_t* bias, int m, int k, int n, int mp, int np, int kp,
                           int scaleBlockNum, int scaleAsymmetric);
}

static constexpr int kSize = 32;
static constexpr int kMeasuredRuns = 5;
static uint64 gAverageCycles = 0;
static uint64 gAverageTicks = 0;
static int gBadResults = -1;
static uint16 gFirstResult = 0;

static int runMockMatMulBenchmark() {
    const size_t activationBytes = kSize * 64 * sizeof(__fp16);
    const size_t outputBytes = kSize * 64 * sizeof(__fp16);
    const size_t weightBytes = kSize * kSize / 2;
    const size_t vectorBytes = kSize * sizeof(__fp16);

    __fp16* activation = (__fp16*)memalign(2048, activationBytes);
    uint8_t* weight = (uint8_t*)memalign(2048, weightBytes);
    __fp16* scale = (__fp16*)memalign(256, vectorBytes);
    __fp16* bias = (__fp16*)memalign(256, vectorBytes);
    __fp16* output = (__fp16*)memalign(2048, outputBytes);
    if (!activation || !weight || !scale || !bias || !output) {
        free(output);
        free(bias);
        free(scale);
        free(weight);
        free(activation);
        return 9;
    }

    // MatMul tensors use NC64 packing. Only the first 32 lanes are logical data.
    memset(activation, 0, activationBytes);
    memset(output, 0, outputBytes);
    for (int row = 0; row < kSize; ++row) {
        for (int col = 0; col < kSize; ++col) {
            activation[row * 64 + col] = (__fp16)1.0f;
        }
    }
    memset(weight, 0x99, weightBytes);  // Each signed q4 value is 1.
    for (int i = 0; i < kSize; ++i) {
        scale[i] = (__fp16)1.0f;
        bias[i] = (__fp16)0.0f;
    }

    int err = vtcm_manager_acquire();
    printf("vtcm_manager_acquire: err=%d\n", err);
    if (err != AEE_SUCCESS) {
        free(output);
        free(bias);
        free(scale);
        free(weight);
        free(activation);
        return 10;
    }

    // Warm up DMA setup, VTCM paths, and the scalar HMX mock once.
    err = hmx_matmulq4fp16_mle32((uint8_t*)output, (const uint8_t*)activation, weight, (const uint8_t*)scale,
                                 (const uint8_t*)bias, kSize, kSize, kSize, 1, 1, 1, 1, 0);
    if (err != AEE_SUCCESS) {
        vtcm_manager_release();
        free(output);
        free(bias);
        free(scale);
        free(weight);
        free(activation);
        return 11;
    }

    uint64 totalCycles = 0;
    uint64 totalTicks = 0;
    for (int run = 0; run < kMeasuredRuns; ++run) {
        memset(output, 0, outputBytes);
        const uint64 tickStart = HAP_perf_get_qtimer_count();
        const uint64 cycleStart = HAP_perf_get_pcycles();
        err = hmx_matmulq4fp16_mle32((uint8_t*)output, (const uint8_t*)activation, weight, (const uint8_t*)scale,
                                     (const uint8_t*)bias, kSize, kSize, kSize, 1, 1, 1, 1, 0);
        totalCycles += HAP_perf_get_pcycles() - cycleStart;
        totalTicks += HAP_perf_get_qtimer_count() - tickStart;
        if (err != AEE_SUCCESS) {
            break;
        }
    }

    int bad = 0;
    if (err == AEE_SUCCESS) {
        for (int row = 0; row < kSize; ++row) {
            for (int col = 0; col < kSize; ++col) {
                uint16 bits = 0;
                __builtin_memcpy(&bits, &output[row * 64 + col], sizeof(bits));
                if (bits != 0x5000) {  // FP16 32.0
                    ++bad;
                }
            }
        }
    }

    gBadResults = bad;
    gFirstResult = ((uint16*)output)[0];
    gAverageCycles = totalCycles / kMeasuredRuns;
    gAverageTicks = totalTicks / kMeasuredRuns;

    vtcm_manager_release();
    free(output);
    free(bias);
    free(scale);
    free(weight);
    free(activation);
    if (err != AEE_SUCCESS) {
        return 12;
    }
    return bad == 0 ? 0 : 13;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    remote_handle64 handle = 0;
    AEEResult err = htp_ops_open("file:///libMNN_htpops_skel.so", &handle);
    printf("htp_ops_open: err=%d handle=0x%llx\n", err, (unsigned long long)handle);
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
    err = htp_ops_get_skel_arch(handle, &arch);
    printf("htp_ops_get_skel_arch: err=%d arch=0x%lx\n", err, (unsigned long)arch);
    if (err != AEE_SUCCESS || arch != 0x79) {
        htp_ops_close(handle);
        return 3;
    }

    const int benchmarkErr = runMockMatMulBenchmark();
    printf("mock_matmul: err=%d bad=%d first=0x%04lx\n", benchmarkErr, gBadResults,
           (unsigned long)gFirstResult);
    printf("mock_matmul average: pcycles=%llu qtimer_ticks=%llu time_ns=%llu\n",
           (unsigned long long)gAverageCycles, (unsigned long long)gAverageTicks,
           (unsigned long long)(gAverageTicks * 625 / 12));

    const AEEResult closeErr = htp_ops_close(handle);
    printf("htp_ops_close: err=%d\n", closeErr);
    if (benchmarkErr != 0) {
        return benchmarkErr;
    }
    return closeErr == AEE_SUCCESS ? 0 : 4;
}
