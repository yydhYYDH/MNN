#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stddef.h>
#include <stdint.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>

#include "dsp/hvx_utils.h"
#include "dsp/worker_pool.h"
#include "htp_command.h"

extern "C" {

static constexpr int32_t kMaxTopK = 64;

static inline bool topk_better(float valueA, int32_t indexA, float valueB, int32_t indexB) {
  return valueA > valueB || (valueA == valueB && indexA < indexB);
}

static inline bool topk_worse(float valueA, int32_t indexA, float valueB, int32_t indexB) {
  return valueA < valueB || (valueA == valueB && indexA > indexB);
}

static inline void topk_sift_down(float* values, int32_t* indices, int32_t size, int32_t start) {
  int32_t pos = start;
  while (true) {
    const int32_t left = pos * 2 + 1;
    if (left >= size) break;
    int32_t child = left;
    const int32_t right = left + 1;
    if (right < size && topk_worse(values[right], indices[right], values[left], indices[left])) {
      child = right;
    }
    if (!topk_worse(values[child], indices[child], values[pos], indices[pos])) break;
    const float oldValue = values[pos];
    const int32_t oldIndex = indices[pos];
    values[pos] = values[child];
    indices[pos] = indices[child];
    values[child] = oldValue;
    indices[child] = oldIndex;
    pos = child;
  }
}

static inline void topk_insert(float* values, int32_t* indices, int32_t* size, int32_t k, float value,
                               int32_t index) {
  if (*size < k) {
    int32_t pos = (*size)++;
    values[pos] = value;
    indices[pos] = index;
    while (pos > 0) {
      const int32_t parent = (pos - 1) / 2;
      if (!topk_worse(values[pos], indices[pos], values[parent], indices[parent])) break;
      const float oldValue = values[parent];
      const int32_t oldIndex = indices[parent];
      values[parent] = values[pos];
      indices[parent] = indices[pos];
      values[pos] = oldValue;
      indices[pos] = oldIndex;
      pos = parent;
    }
  } else if (topk_better(value, index, values[0], indices[0])) {
    values[0] = value;
    indices[0] = index;
    topk_sift_down(values, indices, *size, 0);
  }
}

static inline void topk_store(float* outValues, int32_t* outIndices, int32_t k, int32_t size, float* values,
                              int32_t* indices) {
  for (int32_t i = size; i < k; ++i) {
    outValues[i] = -INFINITY;
    outIndices[i] = INT_MAX;
  }
  for (int32_t pos = k - 1; pos >= 0; --pos) {
    outValues[pos] = values[0];
    outIndices[pos] = indices[0];
    --size;
    if (size == 0) break;
    values[0] = values[size];
    indices[0] = indices[size];
    topk_sift_down(values, indices, size, 0);
  }
}

struct TopKBlockState {
  const __fp16* row;
  int32_t rowSize;
  int32_t k;
  int32_t blockSize;
  int32_t blockCount;
  float* candidateValues;
  int32_t* candidateIndices;
  worker_synctoken_t sync;
  unsigned int nextBlock;
};

static void topk_block_worker(void* opaque, int workerIndex) {
  (void)workerIndex;
  TopKBlockState* state = (TopKBlockState*)opaque;
  while (true) {
    const unsigned int block = worker_pool_atomic_inc_return(&state->nextBlock) - 1;
    if ((int32_t)block >= state->blockCount) break;
    const int32_t begin = (int32_t)block * state->blockSize;
    const int32_t end = begin + state->blockSize < state->rowSize ? begin + state->blockSize : state->rowSize;
    float values[kMaxTopK];
    int32_t indices[kMaxTopK];
    int32_t size = 0;
    for (int32_t i = begin; i < end; ++i) {
      topk_insert(values, indices, &size, state->k, (float)state->row[i], i);
    }
    topk_store(state->candidateValues + (size_t)block * state->k,
               state->candidateIndices + (size_t)block * state->k, state->k, size, values, indices);
  }
  worker_pool_synctoken_jobdone(&state->sync);
}

static bool topk_parallel_row(const __fp16* row, int32_t rowSize, int32_t k, float* outputValues,
                              int32_t* outputIndices) {
  if (g_max_num_workers <= 1 || rowSize < 4096 || k <= 0 || k > kMaxTopK) return false;
  const int32_t blockSize = 2048;
  const int32_t blockCount = (rowSize + blockSize - 1) / blockSize;
  float* candidates = (float*)malloc((size_t)blockCount * k * sizeof(float));
  int32_t* candidateIndices = (int32_t*)malloc((size_t)blockCount * k * sizeof(int32_t));
  if (candidates == nullptr || candidateIndices == nullptr) {
    free(candidates);
    free(candidateIndices);
    return false;
  }
  TopKBlockState state = {row, rowSize, k, blockSize, blockCount, candidates, candidateIndices, {}, 0};
  const int32_t taskCount = blockCount < (int32_t)g_max_num_workers ? blockCount : (int32_t)g_max_num_workers;
  worker_pool_synctoken_init(&state.sync, taskCount);
  worker_pool_job_t job = {topk_block_worker, &state};
  int32_t submitted = 0;
  for (; submitted < taskCount; ++submitted) {
    if (worker_pool_submit(nullptr, job) != 0) {
      for (int32_t missing = submitted; missing < taskCount; ++missing) {
        worker_pool_synctoken_jobdone(&state.sync);
      }
      worker_pool_synctoken_wait(&state.sync);
      free(candidates);
      free(candidateIndices);
      return false;
    }
  }
  worker_pool_synctoken_wait(&state.sync);
  float values[kMaxTopK];
  int32_t indices[kMaxTopK];
  int32_t size = 0;
  for (int32_t block = 0; block < blockCount; ++block) {
    for (int32_t j = 0; j < k; ++j) {
      const int32_t pos = block * k + j;
      topk_insert(values, indices, &size, k, candidates[pos], candidateIndices[pos]);
    }
  }
  topk_store(outputValues, outputIndices, k, size, values, indices);
  free(candidates);
  free(candidateIndices);
  return true;
}

static inline uint16_t htp_ops_topk_fp16_max_bits(const __fp16* src, int32_t size) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  const __fp16* ptr = src;
  int i = 0;

  __fp16 best_scalar = src[0];
  HVX_Vector best_v = Q6_Vh_vsplat_R(((const uint16_t*)src)[0]);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v = vmemu((const HVX_Vector*)ptr);
    best_v = Q6_Vhf_vmax_VhfVhf(best_v, v);
    ptr += vec_len;
  }

  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 64));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 32));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 16));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 8));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 4));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 2));

  __attribute__((aligned(128))) uint16_t tmp[vec_len];
  vmemu((HVX_Vector*)tmp) = best_v;
  uint16_t best_bits = tmp[0];
  best_scalar = *(__fp16*)&best_bits;

  for (; i < size; ++i) {
    const __fp16 value = src[i];
    if (value > best_scalar) {
      best_scalar = value;
      best_bits = ((const uint16_t*)src)[i];
    }
  }
  return best_bits;
}

AEEResult htp_ops_topkv2_k1_fp16(uint8_t* values, uint8_t* indices, uint8_t* input, int32_t rowSize, int32_t rows) {
  if (values == nullptr || indices == nullptr || input == nullptr || rowSize <= 0 || rows <= 0) {
    return -1;
  }
  const __fp16* src = (const __fp16*)input;
  __fp16* valueOut = (__fp16*)values;
  int32_t* indexOut = (int32_t*)indices;
  for (int r = 0; r < rows; ++r) {
    const __fp16* row = src + (size_t)r * rowSize;
    const uint16_t bestBits = htp_ops_topk_fp16_max_bits(row, rowSize);
    const uint16_t* rowBits = (const uint16_t*)row;
    int32_t bestIndex = 0;
    for (int i = 0; i < rowSize; ++i) {
      if (rowBits[i] == bestBits) {
        bestIndex = i;
        break;
      }
    }
    ((uint16_t*)valueOut)[r] = bestBits;
    indexOut[r] = bestIndex;
  }
  return 0;
}

// The sampler normally asks for a small K (currently 40). Keep the heap on
// the DSP stack and avoid materializing or transferring a row-sized index
// array. The heap root is the least useful selected item.
AEEResult htp_ops_topkv2_fp16(uint8_t* values, uint8_t* indices, uint8_t* input,
                              int32_t rowSize, int32_t rows, int32_t k, int32_t outputBytes) {
  constexpr int32_t kMaxTopK = 64;
  if (values == nullptr || indices == nullptr || input == nullptr || rowSize <= 0 || rows <= 0 || k <= 0 ||
      k > rowSize || k > kMaxTopK || (outputBytes != 2 && outputBytes != 4)) {
    return -1;
  }
  const __fp16* src = (const __fp16*)input;
  __fp16* valueOut = (__fp16*)values;
  int32_t* indexOut = (int32_t*)indices;
  for (int r = 0; r < rows; ++r) {
    const __fp16* row = src + (size_t)r * rowSize;
    float parallelValues[kMaxTopK];
    int32_t parallelIndices[kMaxTopK];
    if (topk_parallel_row(row, rowSize, k, parallelValues, parallelIndices)) {
      for (int32_t j = 0; j < k; ++j) {
        if (outputBytes == 4) {
          ((float*)valueOut)[r * k + j] = parallelValues[j];
        } else {
          // Preserve the source FP16 encoding.  Converting the FP16 input to
          // float for the heap and converting it back here is not bit exact
          // on every Hexagon compiler/runtime combination; TopKV2's FP16
          // output contract is to return the selected input values.
          ((uint16_t*)valueOut)[r * k + j] = ((const uint16_t*)&row[parallelIndices[j]])[0];
        }
        indexOut[r * k + j] = parallelIndices[j];
      }
      continue;
    }
    float heapValues[kMaxTopK];
    int32_t heapIndices[kMaxTopK];
    int32_t heapSize = 0;
    // A lower value is worse. For equal values, the larger index is worse;
    // this gives deterministic ordering consistent with the CPU path's
    // original-index preference for ordinary (non-NaN) logits.
    const auto worse = [](float valueA, int32_t indexA, float valueB, int32_t indexB) {
      return valueA < valueB || (valueA == valueB && indexA > indexB);
    };
    const auto better = [](float valueA, int32_t indexA, float valueB, int32_t indexB) {
      return valueA > valueB || (valueA == valueB && indexA < indexB);
    };
    const auto siftDown = [&](int32_t start) {
      int32_t pos = start;
      while (true) {
        const int32_t left = pos * 2 + 1;
        if (left >= heapSize) {
          break;
        }
        int32_t child = left;
        const int32_t right = left + 1;
        if (right < heapSize && worse(heapValues[right], heapIndices[right], heapValues[left], heapIndices[left])) {
          child = right;
        }
        if (!worse(heapValues[child], heapIndices[child], heapValues[pos], heapIndices[pos])) {
          break;
        }
        const float value = heapValues[pos];
        const int32_t index = heapIndices[pos];
        heapValues[pos] = heapValues[child];
        heapIndices[pos] = heapIndices[child];
        heapValues[child] = value;
        heapIndices[child] = index;
        pos = child;
      }
    };

    for (int32_t i = 0; i < rowSize; ++i) {
      const float value = (float)row[i];
      if (heapSize < k) {
        int32_t pos = heapSize++;
        heapValues[pos] = value;
        heapIndices[pos] = i;
        while (pos > 0) {
          const int32_t parent = (pos - 1) / 2;
          if (!worse(heapValues[pos], heapIndices[pos], heapValues[parent], heapIndices[parent])) {
            break;
          }
          const float parentValue = heapValues[parent];
          const int32_t parentIndex = heapIndices[parent];
          heapValues[parent] = heapValues[pos];
          heapIndices[parent] = heapIndices[pos];
          heapValues[pos] = parentValue;
          heapIndices[pos] = parentIndex;
          pos = parent;
        }
      } else if (better(value, i, heapValues[0], heapIndices[0])) {
        heapValues[0] = value;
        heapIndices[0] = i;
        siftDown(0);
      }
    }

    // Pop the least useful element into the back, producing descending order.
    for (int32_t pos = k - 1; pos >= 0; --pos) {
      if (outputBytes == 4) {
        ((float*)valueOut)[r * k + pos] = heapValues[0];
      } else {
        ((uint16_t*)valueOut)[r * k + pos] = ((const uint16_t*)&row[heapIndices[0]])[0];
      }
      indexOut[r * k + pos] = heapIndices[0];
      --heapSize;
      if (heapSize == 0) {
        break;
      }
      heapValues[0] = heapValues[heapSize];
      heapIndices[0] = heapIndices[heapSize];
      siftDown(0);
    }
  }
  return 0;
}

} // extern "C"
