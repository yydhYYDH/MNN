#pragma once

#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048
#define HMX_FP16_SCALE_TABLE_HALF_ENTRIES 128
#define HMX_FP16_SCALE_COLUMN_STRIDE 2

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

static HMX_INLINE_ALWAYS int hmx_fp16_scale_lane_for_column(int col) {
  return col * HMX_FP16_SCALE_COLUMN_STRIDE;
}

static HMX_INLINE_ALWAYS int hmx_fp16_bias_lane_for_column(int col) {
  return col * HMX_FP16_SCALE_COLUMN_STRIDE + 1;
}

static HMX_INLINE_ALWAYS uint16_t hmx_fp16_bits(float value) {
  __fp16 value_h = (__fp16)value;
  uint16_t bits = 0;
  __builtin_memcpy(&bits, &value_h, sizeof(bits));
  return bits;
}

static HMX_INLINE_ALWAYS uint32_t hmx_fp16_scale_bias_word(float scale, float bias) {
  return (uint32_t)hmx_fp16_bits(scale) | ((uint32_t)hmx_fp16_bits(bias) << 16);
}

static HMX_INLINE_ALWAYS HVX_Vector hmx_fp16_scale_bias_splat(float scale, float bias) {
  return Q6_V_vsplat_R((int)hmx_fp16_scale_bias_word(scale, bias));
}

static HMX_INLINE_ALWAYS HVX_Vector hmx_fp16_scale_splat(float scale) {
  return hmx_fp16_scale_bias_splat(scale, 0.0f);
}

static const __fp16* g_mock_scales = NULL;
static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
#ifndef SIMULATOR_MOCK_HMX
  Q6_bias_mxmem2_A((void*)scales);
#else
  g_mock_scales = (const __fp16*)scales;
#endif
}

// Set an aligned 256-byte scale/bias area. Each 32-bit lane is {bias, scale}.
static HMX_INLINE_ALWAYS void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
  HVX_Vector *pv = (HVX_Vector *) out_scales;

  *pv++ = v_scale;
  *pv   = Q6_V_vzero();
}

static HMX_INLINE_ALWAYS void hmx_init_per_column_scale_bias(void *out_scales, const __fp16 *column_scales,
                                                            const __fp16 *column_biases) {
  __fp16 *p = (__fp16 *)out_scales;
  for (int i = 0; i < HMX_FP16_SCALE_TABLE_HALF_ENTRIES; ++i) {
    p[i] = (__fp16)0.0f;
  }
  for (int col = 0; col < HMX_FP16_TILE_N_COLS; ++col) {
    p[hmx_fp16_scale_lane_for_column(col)] = column_scales[col];
    if (column_biases != NULL) {
      p[hmx_fp16_bias_lane_for_column(col)] = column_biases[col];
    }
  }
}

static HMX_INLINE_ALWAYS void hmx_init_per_column_scales(void *out_scales, const __fp16 *column_scales) {
  hmx_init_per_column_scale_bias(out_scales, column_scales, NULL);
}

static const __fp16* g_mock_row_tiles[256];
static const __fp16* g_mock_col_tiles[256];
static size_t g_mock_tile_count = 0;

static __attribute__((noinline)) void hmx_load_tiles_fp16(const __fp16 *row_tiles, const __fp16 *col_tiles,
                                                          size_t n_tiles) {
#ifndef SIMULATOR_MOCK_HMX
  size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  Q6_activation_hf_mxmem_RR_deep((uintptr_t)row_tiles, (int32_t)limit);
  Q6_weight_hf_mxmem_RR((uintptr_t)col_tiles, (int32_t)limit);
#else
  for (size_t t = 0; t < n_tiles && g_mock_tile_count < 256; ++t) {
    g_mock_row_tiles[g_mock_tile_count] = row_tiles + t * HMX_FP16_TILE_N_ELMS;
    g_mock_col_tiles[g_mock_tile_count] = col_tiles + t * HMX_FP16_TILE_N_ELMS;
    ++g_mock_tile_count;
  }
#endif
}

static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
#ifndef SIMULATOR_MOCK_HMX
  Q6_cvt_hf_acc_R(2);
  __builtin_HEXAGON_M8_mxmem(out, 0);
#else
  HVX_Vector scale = Q6_Vh_vsplat_R(0x3c00);
  HVX_Vector bias  = Q6_V_vzero();
  if (g_mock_scales != NULL) {
    HVX_Vector scale_bias = *(const HVX_UVector *) g_mock_scales;
    HVX_Vector dealt      = Q6_Vh_vdeal_Vh(scale_bias);
    scale                 = dealt;
    bias                  = Q6_V_vror_VR(dealt, 64);
  }
  for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
    HVX_VectorPair acc0 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    HVX_VectorPair acc1 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    for (size_t t = 0; t < g_mock_tile_count; ++t) {
      const __fp16 *a = g_mock_row_tiles[t] + (r / 2) * 64;
      for (int k = 0; k < HMX_FP16_TILE_N_COLS; ++k) {
        HVX_Vector weights = Q6_Vh_vdeal_Vh(*(const HVX_UVector *) (g_mock_col_tiles[t] + (k / 2) * 64));
        if (k & 1) {
          weights = Q6_V_vror_VR(weights, 64);
        }
        HVX_Vector a0 = Q6_Vh_vsplat_R(hmx_fp16_bits((float) a[k * 2]));
        HVX_Vector a1 = Q6_Vh_vsplat_R(hmx_fp16_bits((float) a[k * 2 + 1]));
        HVX_VectorPair product0 = Q6_Wqf32_vmpy_VhfVhf(a0, weights);
        HVX_VectorPair product1 = Q6_Wqf32_vmpy_VhfVhf(a1, weights);
        acc0 = Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(acc0), Q6_V_hi_W(product0)),
                                Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(acc0), Q6_V_lo_W(product0)));
        acc1 = Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(acc1), Q6_V_hi_W(product1)),
                                Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(acc1), Q6_V_lo_W(product1)));
      }
    }
    HVX_Vector row0 = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(acc0)),
                                          Q6_Vsf_equals_Vqf32(Q6_V_hi_W(acc0)));
    HVX_Vector row1 = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(acc1)),
                                          Q6_Vsf_equals_Vqf32(Q6_V_hi_W(acc1)));
    row0             = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(row0, scale), bias));
    row1             = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(row1, scale), bias));
    HVX_VectorPair rows = Q6_W_vshuff_VVR(row1, row0, -2);
    *(HVX_UVector *) (out + (r / 2) * 64) = Q6_V_lo_W(rows);
  }
  g_mock_tile_count = 0;
#endif
}

// compute inner product of two vectors of tiles
static HMX_INLINE_ALWAYS void hmx_dot_fp16(__fp16 *out, const __fp16 *row_tiles, const __fp16 *col_tiles,
                                           size_t n_tiles) {
  hmx_load_tiles_fp16(row_tiles, col_tiles, n_tiles);
  hmx_consume_accumulator_fp16(out);
}
