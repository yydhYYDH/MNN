#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>

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

}  // namespace

extern "C" AEEResult htp_ops_linear_attention_gated_delta(
  uint8_t *output, uint8_t *conv_output, const uint8_t *qkv, const uint8_t *gate, const uint8_t *beta,
  const uint8_t *conv_weight, uint8_t *conv_state, uint8_t *recurrent_state, int32_t batch, int32_t conv_dim,
  int32_t sequence, int32_t num_k_heads, int32_t num_v_heads, int32_t head_k_dim, int32_t head_v_dim,
  int32_t conv_kernel, int32_t qkv_c4, int32_t gate_c4, int32_t beta_c4, int32_t output_c4, int32_t weight_c4,
  int32_t use_qk_l2norm, int32_t c4_pack) {
  if (output == nullptr || conv_output == nullptr || qkv == nullptr || gate == nullptr || beta == nullptr ||
      conv_weight == nullptr || conv_state == nullptr || recurrent_state == nullptr || batch <= 0 || conv_dim <= 0 ||
      sequence <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0 || conv_kernel <= 0 ||
      num_v_heads % num_k_heads != 0 || c4_pack <= 0) {
    return AEE_EBADPARM;
  }

  const int conv_state_size = conv_kernel - 1;
  const int key_dim         = num_k_heads * head_k_dim;
  const int value_dim       = num_v_heads * head_v_dim;
  if (conv_dim != 2 * key_dim + value_dim) {
    return AEE_EBADPARM;
  }
  const int   gqa             = num_v_heads / num_k_heads;
  const float q_scale         = 1.0f / sqrtf((float) head_k_dim);
  const int   output_channels = value_dim;
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < sequence; ++t) {
      // Depthwise causal convolution followed by SiLU. State is kept in plain
      // channel-major layout so the recurrent command can update it in place.
      for (int d = 0; d < conv_dim; ++d) {
        float sum = 0.0f;
        for (int k = 0; k < conv_kernel; ++k) {
          float x = 0.0f;
          if (k < conv_state_size) {
            const int state_index = ((b * conv_dim + d) * conv_state_size) + k;
            x                     = load_fp16(conv_state, state_index);
          } else {
            x = load_sequence(qkv, b, d, t, batch, conv_dim, sequence, qkv_c4, c4_pack);
          }
          const int weight_index = weight_c4 ? c4_offset(k, d, conv_kernel, c4_pack) : d * conv_kernel + k;
          sum += x * load_fp16(conv_weight, weight_index);
        }
        const float silu           = sum / (1.0f + expf(-sum));
        const int   conv_out_index = ((b * sequence + t) * conv_dim) + d;
        // Shift the causal state and append the current input.
        for (int k = 0; k + 1 < conv_state_size; ++k) {
          const int dst = (b * conv_dim + d) * conv_state_size + k;
          store_fp16(conv_state, dst, load_fp16(conv_state, dst + 1));
        }
        if (conv_state_size > 0) {
          store_fp16(conv_state, (b * conv_dim + d) * conv_state_size + conv_state_size - 1,
                     load_sequence(qkv, b, d, t, batch, conv_dim, sequence, qkv_c4, c4_pack));
        }

        store_fp16(conv_output, conv_out_index, silu);
      }

      for (int h = 0; h < num_v_heads; ++h) {
        const int kh = h / gqa;
        float     q[256];
        float     k[256];
        if (head_k_dim > 256 || head_v_dim > 256) {
          return AEE_EBADPARM;
        }
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) {
          const int q_index = (b * sequence + t) * conv_dim + kh * head_k_dim + i;
          const int k_index = (b * sequence + t) * conv_dim + key_dim + kh * head_k_dim + i;
          q[i]              = load_fp16(conv_output, q_index);
          k[i]              = load_fp16(conv_output, k_index);
          q_norm += q[i] * q[i];
          k_norm += k[i] * k[i];
        }
        const float q_factor = use_qk_l2norm ? q_scale / sqrtf(q_norm + 1.0e-6f) : 1.0f;
        const float k_factor = use_qk_l2norm ? 1.0f / sqrtf(k_norm + 1.0e-6f) : 1.0f;
        float       dot      = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) {
          q[i] *= q_factor;
          k[i] *= k_factor;
          dot += q[i] * k[i];
        }

        const int   state_base = ((b * num_v_heads + h) * head_k_dim) * head_v_dim;
        const float decay = expf(load_token_channel(gate, b, t, h, batch, num_v_heads, sequence, gate_c4, c4_pack));
        const float beta_value = load_token_channel(beta, b, t, h, batch, num_v_heads, sequence, beta_c4, c4_pack);
        for (int j = 0; j < head_v_dim; ++j) {
          float predicted = 0.0f;
          float queried   = 0.0f;
          for (int i = 0; i < head_k_dim; ++i) {
            const float state = load_fp16(recurrent_state, state_base + i * head_v_dim + j);
            predicted += state * k[i];
            queried += state * q[i];
          }
          const int   v_index = (b * sequence + t) * conv_dim + 2 * key_dim + h * head_v_dim + j;
          const float value   = load_fp16(conv_output, v_index);
          const float delta   = beta_value * (value - decay * predicted);
          const float result  = decay * queried + dot * delta;
          if (output_c4) {
            const int token = (b * sequence + t) * num_v_heads + h;
            store_fp16(output, c4_offset(token, j, batch * sequence * num_v_heads, c4_pack), result);
          } else {
            store_output(output, b, t, h * head_v_dim + j, batch, output_channels, sequence, 0, c4_pack, result);
          }
          for (int i = 0; i < head_k_dim; ++i) {
            const int state_index = state_base + i * head_v_dim + j;
            store_fp16(recurrent_state, state_index, decay * load_fp16(recurrent_state, state_index) + k[i] * delta);
          }
        }
      }
    }
  }
  return AEE_SUCCESS;
}
