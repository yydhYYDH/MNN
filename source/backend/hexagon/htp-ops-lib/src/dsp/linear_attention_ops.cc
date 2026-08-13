#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>

#include "attention_private.hpp"
#include "dsp/pwl.h"

namespace {

static inline int c4_offset(int token, int channel, int token_count, int pack) {
  return (channel / pack) * token_count * pack + token * pack + channel % pack;
}

static inline float load_fp16(const uint8_t *data, int index) {
  return (float) ((const __fp16 *) data)[index];
}

static inline void store_fp16(uint8_t *data, int index, float value) {
  ((__fp16 *) data)[index] = (__fp16) value;
}

static inline int32_t fp32_bits(float value) {
  union { float f; int32_t i; } bits = { .f = value };
  return bits.i;
}

static inline float load_sequence(const uint8_t *data, int b, int d, int t, int batch, int channels, int sequence,
                                  int c4, int pack) {
  const int index = c4 ? c4_offset(b * sequence + t, d, batch * sequence, pack) : (b * channels + d) * sequence + t;
  return load_fp16(data, index);
}

static inline float load_token_channel(const uint8_t *data, int b, int t, int c, int batch, int channels, int sequence,
                                       int c4, int pack) {
  const int index = c4 ? c4_offset(b * sequence + t, c, batch * sequence, pack) : (b * sequence + t) * channels + c;
  return load_fp16(data, index);
}

static inline void store_output(uint8_t *data, int b, int t, int c, int batch, int channels, int sequence, int c4,
                                int pack, float value) {
  const int index = c4 ? c4_offset(b * sequence + t, c, batch * sequence, pack) : (b * sequence + t) * channels + c;
  store_fp16(data, index, value);
}

struct LinearAttentionHmxJob {
  const __fp16 *q;
  const __fp16 *k;
  const __fp16 *state;
  __fp16       *output;
  int           head_k_dim;
  int           head_v_dim;
};

static inline void linear_attention_hmx_pack_activation(__fp16 *dst, const __fp16 *q, const __fp16 *k, int head_k_dim) {
  const int k_tiles = head_k_dim / 32;
  memset(dst, 0, (size_t) k_tiles * 1024 * sizeof(__fp16));
  for (int kt = 0; kt < k_tiles; ++kt) {
    __fp16 *tile = dst + (size_t) kt * 1024;
    for (int col = 0; col < 32; ++col) {
      tile[col * 2]     = q[kt * 32 + col];
      tile[col * 2 + 1] = k[kt * 32 + col];
    }
  }
}

static inline size_t linear_attention_packed_state_index(int row, int col, int head_k_dim, int head_v_dim) {
  const int k_tiles = head_k_dim / 32;
  const int kt      = row / 32;
  const int nt      = col / 32;
  return ((size_t) nt * k_tiles + kt) * 1024 + ((row & 31) / 2) * 64 + (col & 31) * 2 + (row & 1);
}

static inline void linear_attention_copy_packed_state(__fp16 *dst, const __fp16 *src, size_t bytes) {
  memcpy(dst, src, bytes);
}

static inline void linear_attention_hmx_store_two_rows(__fp16 *dst, const __fp16 *tile, int col_offset,
                                                       int head_v_dim) {
  for (int col = 0; col < 32; ++col) {
    dst[col_offset + col]              = tile[col * 2];
    dst[head_v_dim + col_offset + col] = tile[col * 2 + 1];
  }
}

static void linear_attention_hmx_job(void *opaque) {
  LinearAttentionHmxJob *job        = (LinearAttentionHmxJob *) opaque;
  const int              k_tiles    = job->head_k_dim / 32;
  const int              n_tiles    = job->head_v_dim / 32;
  uint8_t               *vtcm       = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16                *activation = (__fp16 *) vtcm_seq_alloc(&vtcm, (size_t) k_tiles * 1024 * sizeof(__fp16));
  __fp16                *weight = (__fp16 *) vtcm_seq_alloc(&vtcm, (size_t) n_tiles * k_tiles * 1024 * sizeof(__fp16));
  __fp16                *tile_output = (__fp16 *) vtcm_seq_alloc(&vtcm, 1024 * sizeof(__fp16));
  __fp16                *scales      = (__fp16 *) vtcm_seq_alloc(&vtcm, 256);

  linear_attention_hmx_pack_activation(activation, job->q, job->k, job->head_k_dim);
  linear_attention_copy_packed_state(weight, job->state, (size_t) job->head_k_dim * job->head_v_dim * sizeof(__fp16));
  hmx_init_column_scales(scales, hmx_fp16_scale_splat(1.0f));
  hmx_set_output_scales(scales);
  for (int nt = 0; nt < n_tiles; ++nt) {
    hmx_load_tiles_fp16(activation, weight + (size_t) nt * k_tiles * 1024, k_tiles);
    hmx_consume_accumulator_fp16(tile_output);
    linear_attention_hmx_store_two_rows(job->output, tile_output, nt * 32, job->head_v_dim);
  }
}

static bool linear_attention_hmx_matmul(__fp16 *output, const __fp16 *q, const __fp16 *k, const __fp16 *state,
                                        int head_k_dim, int head_v_dim) {
  if (head_k_dim <= 0 || head_v_dim <= 0 || head_k_dim > 256 || head_v_dim > 256 || (head_k_dim & 31) != 0 ||
      (head_v_dim & 31) != 0) {
    return false;
  }
  LinearAttentionHmxJob job = { q, k, state, output, head_k_dim, head_v_dim };
  hmx_queue_execute_with_spin(linear_attention_hmx_job, &job, 2000);
  return true;
}

static inline bool linear_attention_uses_packed_state(int head_k_dim, int head_v_dim) {
  return head_k_dim > 0 && head_v_dim > 0 && head_k_dim <= 256 && head_v_dim <= 256 && (head_k_dim & 31) == 0 &&
         (head_v_dim & 31) == 0;
}

static inline void linear_attention_hvx_update_packed_state(__fp16 *state, const float *k, const __fp16 *delta,
                                                            float decay, int head_k_dim, int head_v_dim) {
  const int k_tiles = head_k_dim / 32;
  const int n_tiles = head_v_dim / 32;
  // Scalar fp32 accumulation: state = decay * state + k * delta in fp32,
  // stored back as fp16. This avoids per-token fp16 rounding drift in the
  // recurrent state that accumulates over long sequences.
  for (int kt = 0; kt < k_tiles; ++kt) {
    for (int row_pair = 0; row_pair < 16; ++row_pair) {
      const int row0 = kt * 32 + row_pair * 2;
      // The packed layout pairs two k rows (row0, row0+1) with the 64-lane
      // state tile: low half = k[row0], high half = k[row0+1] for every lane.
      const float k0 = k[row0];
      const float k1 = k[row0 + 1];
      for (int nt = 0; nt < n_tiles; ++nt) {
        const size_t tile_base = ((size_t) nt * k_tiles + kt) * 1024 + row_pair * 64;
        __fp16 *s = state + tile_base;
        for (int col = 0; col < 32; ++col) {
          const float st0 = (float)s[col * 2];
          const float st1 = (float)s[col * 2 + 1];
          const float d   = (float)delta[nt * 32 + col];
          s[col * 2]     = (__fp16)(decay * st0 + k0 * d);
          s[col * 2 + 1] = (__fp16)(decay * st1 + k1 * d);
        }
      }
    }
  }
}

struct LinearAttentionConvTask {
  uint8_t            *conv_output;
  const uint8_t      *qkv;
  const uint8_t      *conv_weight;
  uint8_t            *conv_state;
  int                 b;
  int                 t;
  int                 batch;
  int                 conv_dim;
  int                 sequence;
  int                 conv_kernel;
  int                 qkv_c4;
  int                 weight_c4;
  int                 c4_pack;
  int                 channel_begin;
  int                 channel_end;
  worker_synctoken_t *sync;
};

static void linear_attention_convolution_range(const LinearAttentionConvTask *task) {
  const int conv_state_size = task->conv_kernel - 1;
  if (task->batch == 1 && task->conv_kernel == 4 && task->qkv_c4 && task->c4_pack == 64 &&
      (task->channel_begin & 63) == 0 && (task->channel_end & 63) == 0) {
    __fp16       *state  = (__fp16 *) task->conv_state;
    const __fp16 *qkv    = (const __fp16 *) task->qkv;
    const __fp16 *weight = (const __fp16 *) task->conv_weight;
    __fp16       *output = (__fp16 *) task->conv_output;
    for (int d = task->channel_begin; d < task->channel_end; d += 64) {
      const int        block = d / 64;
      const HVX_Vector x0    = vmemu(state + d);
      const HVX_Vector x1    = vmemu(state + task->conv_dim + d);
      const HVX_Vector x2    = vmemu(state + 2 * task->conv_dim + d);
      const HVX_Vector x3    = vmemu(qkv + ((size_t) block * task->sequence + task->t) * 64);
      _Alignas(128) __fp16 plain_weights[4][64];
      HVX_Vector weights[4];
      if (task->weight_c4) {
        const __fp16 *weight_block = weight + (size_t) block * task->conv_kernel * 64;
        for (int tap = 0; tap < 4; ++tap) {
          weights[tap] = vmemu(weight_block + tap * 64);
        }
      } else {
        for (int lane = 0; lane < 64; ++lane) {
          for (int tap = 0; tap < 4; ++tap) {
            plain_weights[tap][lane] = weight[(size_t) (d + lane) * 4 + tap];
          }
        }
        for (int tap = 0; tap < 4; ++tap) {
          weights[tap] = vmem(plain_weights[tap]);
        }
      }
      HVX_Vector sum = Q6_Vhf_vmpy_VhfVhf(x0, weights[0]);
      sum            = Q6_Vhf_vmpyacc_VhfVhfVhf(sum, x1, weights[1]);
      sum            = Q6_Vhf_vmpyacc_VhfVhfVhf(sum, x2, weights[2]);
      sum            = Q6_Vhf_vmpyacc_VhfVhfVhf(sum, x3, weights[3]);
      const HVX_Vector activated = htp_ops_silu_pwl_fp16_vec(sum);
      vmemu(output + (size_t) (task->b * task->sequence + task->t) * task->conv_dim + d) = activated;
      vmemu(state + d)                      = x1;
      vmemu(state + task->conv_dim + d)     = x2;
      vmemu(state + 2 * task->conv_dim + d) = x3;
    }
    return;
  }
  for (int d = task->channel_begin; d < task->channel_end; ++d) {
    float sum = 0.0f;
    for (int k = 0; k < task->conv_kernel; ++k) {
      float x = 0.0f;
      if (k < conv_state_size) {
        x = load_fp16(task->conv_state, ((task->b * task->conv_dim + d) * conv_state_size) + k);
      } else {
        x = load_sequence(task->qkv, task->b, d, task->t, task->batch, task->conv_dim, task->sequence, task->qkv_c4,
                          task->c4_pack);
      }
      const int weight_index =
        task->weight_c4 ? c4_offset(k, d, task->conv_kernel, task->c4_pack) : d * task->conv_kernel + k;
      sum += x * load_fp16(task->conv_weight, weight_index);
    }
    for (int k = 0; k + 1 < conv_state_size; ++k) {
      const int dst = (task->b * task->conv_dim + d) * conv_state_size + k;
      store_fp16(task->conv_state, dst, load_fp16(task->conv_state, dst + 1));
    }
    if (conv_state_size > 0) {
      store_fp16(task->conv_state, (task->b * task->conv_dim + d) * conv_state_size + conv_state_size - 1,
                 load_sequence(task->qkv, task->b, d, task->t, task->batch, task->conv_dim, task->sequence,
                               task->qkv_c4, task->c4_pack));
    }
    store_fp16(task->conv_output, (task->b * task->sequence + task->t) * task->conv_dim + d, sum / (1.0f + expf(-sum)));
  }
}

static void linear_attention_convolution_worker(void *opaque, int worker_index) {
  (void) worker_index;
  LinearAttentionConvTask *task = (LinearAttentionConvTask *) opaque;
  linear_attention_convolution_range(task);
  worker_pool_synctoken_jobdone(task->sync);
}

static void linear_attention_convolution_token(uint8_t *conv_output, const uint8_t *qkv, const uint8_t *conv_weight,
                                               uint8_t *conv_state, int b, int t, int batch, int conv_dim, int sequence,
                                               int conv_kernel, int qkv_c4, int weight_c4, int c4_pack) {
  int task_count = g_max_num_workers > 1 && conv_dim >= 1024 ? (int) g_max_num_workers : 1;
  if (task_count > 4) {
    task_count = 4;
  }
  if (task_count <= 1) {
    LinearAttentionConvTask task = { conv_output, qkv,         conv_weight, conv_state, b,       t, batch,    conv_dim,
                                     sequence,    conv_kernel, qkv_c4,      weight_c4,  c4_pack, 0, conv_dim, nullptr };
    linear_attention_convolution_range(&task);
    return;
  }
  LinearAttentionConvTask *tasks = WORKER_POOL_STACK_ALLOC(LinearAttentionConvTask, task_count);
  worker_synctoken_t       sync;
  worker_pool_synctoken_init(&sync, task_count);
  const int channels_per_task = (conv_dim + task_count - 1) / task_count;
  for (int i = 0; i < task_count; ++i) {
    const int begin = i * channels_per_task;
    int       end   = begin + channels_per_task;
    if (end > conv_dim) {
      end = conv_dim;
    }
    tasks[i]              = { conv_output, qkv,         conv_weight, conv_state, b,       t,     batch, conv_dim,
                              sequence,    conv_kernel, qkv_c4,      weight_c4,  c4_pack, begin, end,   &sync };
    worker_pool_job_t job = { linear_attention_convolution_worker, tasks + i };
    worker_pool_submit(nullptr, job);
  }
  worker_pool_synctoken_wait(&sync);
}

struct LinearAttentionHeadTask {
  uint8_t            *output;
  const uint8_t      *conv_output;
  const uint8_t      *gate;
  const uint8_t      *beta;
  uint8_t            *recurrent_state;
  int                 b;
  int                 t;
  int                 batch;
  int                 conv_dim;
  int                 sequence;
  int                 num_k_heads;
  int                 num_v_heads;
  int                 head_k_dim;
  int                 head_v_dim;
  int                 gate_c4;
  int                 beta_c4;
  int                 output_c4;
  int                 use_qk_l2norm;
  int                 c4_pack;
  int                 head_begin;
  int                 head_end;
  worker_synctoken_t *sync;
};

static void linear_attention_head_range(const LinearAttentionHeadTask *task) {
  const int   key_dim   = task->num_k_heads * task->head_k_dim;
  const int   value_dim = task->num_v_heads * task->head_v_dim;
  const int   gqa       = task->num_v_heads / task->num_k_heads;
  const float q_scale   = 1.0f / sqrtf((float) task->head_k_dim);
  for (int h = task->head_begin; h < task->head_end; ++h) {
    const int kh = h / gqa;
    _Alignas(128) float q[256];
    _Alignas(128) float k[256];
    _Alignas(128) __fp16 qk_fp16[512];
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    const int q_base = (task->b * task->sequence + task->t) * task->conv_dim + kh * task->head_k_dim;
    const int k_base = q_base + key_dim;
    const bool hvx_qk = (task->head_k_dim & 63) == 0;
    if (hvx_qk) {
      _Alignas(128) float q_sq[256];
      _Alignas(128) float k_sq[256];
      for (int i = 0; i < task->head_k_dim; i += 64) {
        const HVX_VectorPair q_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(
          vmemu((const __fp16 *) task->conv_output + q_base + i)));
        const HVX_VectorPair k_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(
          vmemu((const __fp16 *) task->conv_output + k_base + i)));
        vmemu(q + i) = Q6_V_lo_W(q_sf);
        vmemu(q + i + 32) = Q6_V_hi_W(q_sf);
        vmemu(k + i) = Q6_V_lo_W(k_sf);
        vmemu(k + i + 32) = Q6_V_hi_W(k_sf);
        vmemu(q_sq + i) = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(q_sf), Q6_V_lo_W(q_sf));
        vmemu(q_sq + i + 32) = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(q_sf), Q6_V_hi_W(q_sf));
        vmemu(k_sq + i) = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(k_sf), Q6_V_lo_W(k_sf));
        vmemu(k_sq + i + 32) = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(k_sf), Q6_V_hi_W(k_sf));
      }
      for (int i = 0; i < task->head_k_dim; ++i) {
        q_norm += q_sq[i];
        k_norm += k_sq[i];
      }
    } else for (int i = 0; i < task->head_k_dim; ++i) {
      const int q_index = (task->b * task->sequence + task->t) * task->conv_dim + kh * task->head_k_dim + i;
      const int k_index = (task->b * task->sequence + task->t) * task->conv_dim + key_dim + kh * task->head_k_dim + i;
      q[i]              = load_fp16(task->conv_output, q_index);
      k[i]              = load_fp16(task->conv_output, k_index);
      q_norm += q[i] * q[i];
      k_norm += k[i] * k[i];
    }
    const float q_factor = task->use_qk_l2norm ? q_scale / sqrtf(q_norm + 1.0e-6f) : 1.0f;
    const float k_factor = task->use_qk_l2norm ? 1.0f / sqrtf(k_norm + 1.0e-6f) : 1.0f;
    float       dot      = 0.0f;
    if (hvx_qk) {
      _Alignas(128) float qk_products[256];
      const HVX_Vector v_q_factor = Q6_V_vsplat_R(fp32_bits(q_factor));
      const HVX_Vector v_k_factor = Q6_V_vsplat_R(fp32_bits(k_factor));
      for (int i = 0; i < task->head_k_dim; i += 64) {
        const HVX_Vector q0 = Q6_Vsf_vmpy_VsfVsf(vmemu(q + i), v_q_factor);
        const HVX_Vector q1 = Q6_Vsf_vmpy_VsfVsf(vmemu(q + i + 32), v_q_factor);
        const HVX_Vector k0 = Q6_Vsf_vmpy_VsfVsf(vmemu(k + i), v_k_factor);
        const HVX_Vector k1 = Q6_Vsf_vmpy_VsfVsf(vmemu(k + i + 32), v_k_factor);
        vmemu(q + i) = q0;
        vmemu(q + i + 32) = q1;
        vmemu(k + i) = k0;
        vmemu(k + i + 32) = k1;
        vmemu(qk_products + i) = Q6_Vsf_vmpy_VsfVsf(q0, k0);
        vmemu(qk_products + i + 32) = Q6_Vsf_vmpy_VsfVsf(q1, k1);
        vmemu(qk_fp16 + i) = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(q0, q1));
        vmemu(qk_fp16 + task->head_k_dim + i) = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(k0, k1));
      }
      for (int i = 0; i < task->head_k_dim; ++i) {
        dot += qk_products[i];
      }
    } else for (int i = 0; i < task->head_k_dim; ++i) {
      q[i] *= q_factor;
      k[i] *= k_factor;
      qk_fp16[i]                    = (__fp16) q[i];
      qk_fp16[task->head_k_dim + i] = (__fp16) k[i];
      dot += q[i] * k[i];
    }

    const int   state_base = ((task->b * task->num_v_heads + h) * task->head_k_dim) * task->head_v_dim;
    const float decay      = expf(load_token_channel(task->gate, task->b, task->t, h, task->batch, task->num_v_heads,
                                                     task->sequence, task->gate_c4, task->c4_pack));
    const float beta_value = load_token_channel(task->beta, task->b, task->t, h, task->batch, task->num_v_heads,
                                                task->sequence, task->beta_c4, task->c4_pack);
    __fp16 hmx_products[512];
    const bool packed_state = linear_attention_uses_packed_state(task->head_k_dim, task->head_v_dim);
    const bool use_hmx      = linear_attention_hmx_matmul(hmx_products, qk_fp16, qk_fp16 + task->head_k_dim,
                                                          (const __fp16 *) task->recurrent_state + state_base,
                                                          task->head_k_dim, task->head_v_dim);
    _Alignas(128) __fp16 delta_values[256];
    const bool hvx_delta_output = use_hmx && task->output_c4 && (task->head_v_dim & 63) == 0;
    if (hvx_delta_output) {
      const HVX_Vector v_decay = Q6_V_vsplat_R(fp32_bits(decay));
      const HVX_Vector v_beta  = Q6_V_vsplat_R(fp32_bits(beta_value));
      const HVX_Vector v_dot   = Q6_V_vsplat_R(fp32_bits(dot));
      const int token = (task->b * task->sequence + task->t) * task->num_v_heads + h;
      for (int j = 0; j < task->head_v_dim; j += 64) {
        const int v_index = (task->b * task->sequence + task->t) * task->conv_dim + 2 * key_dim +
                            h * task->head_v_dim + j;
        const HVX_VectorPair v_value = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(
          vmemu((const __fp16 *) task->conv_output + v_index)));
        const HVX_VectorPair v_predicted = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(
          vmemu(hmx_products + task->head_v_dim + j)));
        const HVX_VectorPair v_queried = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vmemu(hmx_products + j)));
        HVX_Vector delta_sf[2];
        HVX_Vector result_sf[2];
        for (int half = 0; half < 2; ++half) {
          const HVX_Vector value_sf = half ? Q6_V_hi_W(v_value) : Q6_V_lo_W(v_value);
          const HVX_Vector predicted_sf = half ? Q6_V_hi_W(v_predicted) : Q6_V_lo_W(v_predicted);
          const HVX_Vector queried_sf = half ? Q6_V_hi_W(v_queried) : Q6_V_lo_W(v_queried);
          const HVX_Vector residual_sf = Q6_Vsf_vsub_VsfVsf(
            value_sf, Q6_Vsf_vmpy_VsfVsf(v_decay, predicted_sf));
          delta_sf[half] = Q6_Vsf_vmpy_VsfVsf(v_beta, residual_sf);
          result_sf[half] = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v_decay, queried_sf),
                                               Q6_Vsf_vmpy_VsfVsf(v_dot, delta_sf[half]));
        }
        const HVX_Vector v_delta = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(delta_sf[0], delta_sf[1]));
        const HVX_Vector v_result = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(result_sf[0], result_sf[1]));
        vmemu(delta_values + j) = v_delta;
        vmemu((__fp16 *) task->output + c4_offset(token, j, task->batch * task->sequence * task->num_v_heads,
                                                  task->c4_pack)) = v_result;
      }
    } else for (int j = 0; j < task->head_v_dim; ++j) {
      float predicted = use_hmx ? (float) hmx_products[task->head_v_dim + j] : 0.0f;
      float queried   = use_hmx ? (float) hmx_products[j] : 0.0f;
      if (!use_hmx) {
        for (int i = 0; i < task->head_k_dim; ++i) {
          const size_t state_offset = packed_state ?
                                        linear_attention_packed_state_index(i, j, task->head_k_dim, task->head_v_dim) :
                                        (size_t) i * task->head_v_dim + j;
          const float  state        = load_fp16(task->recurrent_state, state_base + state_offset);
          predicted += state * k[i];
          queried += state * q[i];
        }
      }
      const int v_index =
        (task->b * task->sequence + task->t) * task->conv_dim + 2 * key_dim + h * task->head_v_dim + j;
      const float value  = load_fp16(task->conv_output, v_index);
      const float delta  = beta_value * (value - decay * predicted);
      delta_values[j]    = (__fp16) delta;
      const float result = decay * queried + dot * delta;
      if (task->output_c4) {
        const int token = (task->b * task->sequence + task->t) * task->num_v_heads + h;
        store_fp16(task->output, c4_offset(token, j, task->batch * task->sequence * task->num_v_heads, task->c4_pack),
                   result);
      } else {
        store_output(task->output, task->b, task->t, h * task->head_v_dim, task->batch, value_dim, task->sequence, 0,
                     task->c4_pack, result);
      }
    }
    if (packed_state) {
      linear_attention_hvx_update_packed_state((__fp16 *) task->recurrent_state + state_base, k, delta_values, decay,
                                               task->head_k_dim, task->head_v_dim);
    } else {
      for (int j = 0; j < task->head_v_dim; ++j) {
        for (int i = 0; i < task->head_k_dim; ++i) {
          const size_t state_index = state_base + (size_t) i * task->head_v_dim + j;
          store_fp16(task->recurrent_state, state_index,
                     decay * load_fp16(task->recurrent_state, state_index) + k[i] * delta_values[j]);
        }
      }
    }
  }
}

static void linear_attention_head_worker(void *opaque, int worker_index) {
  (void) worker_index;
  LinearAttentionHeadTask *task = (LinearAttentionHeadTask *) opaque;
  linear_attention_head_range(task);
  worker_pool_synctoken_jobdone(task->sync);
}

static void linear_attention_heads_token(const LinearAttentionHeadTask &base) {
  int task_count = g_max_num_workers > 1 && base.num_v_heads > 1 ? (int) g_max_num_workers : 1;
  if (task_count > base.num_v_heads) {
    task_count = base.num_v_heads;
  }
  if (task_count <= 1) {
    LinearAttentionHeadTask task = base;
    task.head_begin              = 0;
    task.head_end                = base.num_v_heads;
    linear_attention_head_range(&task);
    return;
  }
  LinearAttentionHeadTask *tasks = WORKER_POOL_STACK_ALLOC(LinearAttentionHeadTask, task_count);
  worker_synctoken_t       sync;
  worker_pool_synctoken_init(&sync, task_count);
  const int heads_per_task = (base.num_v_heads + task_count - 1) / task_count;
  for (int i = 0; i < task_count; ++i) {
    tasks[i]            = base;
    tasks[i].head_begin = i * heads_per_task;
    tasks[i].head_end   = tasks[i].head_begin + heads_per_task;
    if (tasks[i].head_end > base.num_v_heads) {
      tasks[i].head_end = base.num_v_heads;
    }
    tasks[i].sync         = &sync;
    worker_pool_job_t job = { linear_attention_head_worker, tasks + i };
    worker_pool_submit(nullptr, job);
  }
  worker_pool_synctoken_wait(&sync);
}

}  // namespace

extern "C" AEEResult htp_ops_linear_attention_gated_delta(
  uint8_t *output, uint8_t *conv_output, const uint8_t *qkv, const uint8_t *gate, const uint8_t *beta,
  const uint8_t *conv_weight, uint8_t *conv_state, uint8_t *recurrent_state, uint8_t *packed_conv_weight,
  int32_t batch, int32_t conv_dim,
  int32_t sequence, int32_t num_k_heads, int32_t num_v_heads, int32_t head_k_dim, int32_t head_v_dim,
  int32_t conv_kernel, int32_t qkv_c4, int32_t gate_c4, int32_t beta_c4, int32_t output_c4, int32_t weight_c4,
  int32_t use_qk_l2norm, int32_t c4_pack) {
  if (output == nullptr || conv_output == nullptr || qkv == nullptr || gate == nullptr || beta == nullptr ||
      conv_weight == nullptr || conv_state == nullptr || recurrent_state == nullptr || packed_conv_weight == nullptr ||
      batch <= 0 || conv_dim <= 0 ||
      sequence <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0 || conv_kernel <= 0 ||
      head_k_dim > 256 || head_v_dim > 256 || num_v_heads % num_k_heads != 0 || c4_pack <= 0) {
    return AEE_EBADPARM;
  }

  const int key_dim   = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  if (conv_dim != 2 * key_dim + value_dim) {
    return AEE_EBADPARM;
  }
  constexpr uint32_t kPackedConvWeightMagic = 0x4c415743;
  constexpr int kPackedConvWeightHeaderBytes = 128;
  const bool use_packed_conv_weight = batch == 1 && conv_kernel == 4 && c4_pack == 64 && (conv_dim & 63) == 0;
  if (use_packed_conv_weight && ((uint32_t *) packed_conv_weight)[0] != kPackedConvWeightMagic) {
    __fp16 *packed = (__fp16 *) (packed_conv_weight + kPackedConvWeightHeaderBytes);
    const __fp16 *weight = (const __fp16 *) conv_weight;
    for (int block = 0; block < conv_dim / 64; ++block) {
      for (int tap = 0; tap < conv_kernel; ++tap) {
        for (int lane = 0; lane < 64; ++lane) {
          const int channel = block * 64 + lane;
          const int src = weight_c4 ? c4_offset(tap, channel, conv_kernel, c4_pack)
                                    : channel * conv_kernel + tap;
          packed[((size_t) block * conv_kernel + tap) * 64 + lane] = weight[src];
        }
      }
    }
    ((uint32_t *) packed_conv_weight)[0] = kPackedConvWeightMagic;
  }
  const uint8_t *command_conv_weight = use_packed_conv_weight
      ? packed_conv_weight + kPackedConvWeightHeaderBytes : conv_weight;
  const int command_weight_c4 = use_packed_conv_weight ? 1 : weight_c4;
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < sequence; ++t) {
      linear_attention_convolution_token(conv_output, qkv, command_conv_weight, conv_state, b, t, batch, conv_dim,
                                         sequence, conv_kernel, qkv_c4, command_weight_c4, c4_pack);

      LinearAttentionHeadTask head_task = { output,      conv_output, gate,          beta,       recurrent_state,
                                            b,           t,           batch,         conv_dim,   sequence,
                                            num_k_heads, num_v_heads, head_k_dim,    head_v_dim, gate_c4,
                                            beta_c4,     output_c4,   use_qk_l2norm, c4_pack,    0,
                                            num_v_heads, nullptr };
      linear_attention_heads_token(head_task);
    }
  }
  return AEE_SUCCESS;
}
