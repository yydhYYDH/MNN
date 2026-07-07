# Qwen3-VL MNN Alignment Notes

Date: 2026-07-06

## Scope

This note records the patches and findings from comparing Torch/HF, vLLM-style input, and MNN runtime for the Qwen3-VL UI-Venus model.

## Patches Applied

1. C++ runtime resize interpolation alignment

   Added a Qwen3-VL Pillow-like BICUBIC resize path in `transformers/llm/engine/src/omni.cpp`.

   Purpose: make MNN runtime image pixel preprocessing match HF/Pillow more closely. MNN previously used:

   ```cpp
   MNN::CV::resize(..., MNN::CV::INTER_LINEAR, ...)
   ```

   for Qwen3-VL preprocessing.

   Current Qwen3-VL path calls:

   ```cpp
   qwen3PillowLikeResize(image, mVisionWidth, mVisionHeight)
   ```

   It is enabled by default through:

   ```cpp
   qwen3_vl_pillow_resize = true
   ```

   Result: Qwen3-VL resize interpolation now follows a Pillow-like BICUBIC path instead of MNN linear interpolation.

2. C++ runtime `smart_resize` size alignment

   Replaced the previous simple alignment:

   ```cpp
   round(h / factor) * factor
   ```

   with HF-equivalent `smart_resize` logic:

   - Python 3 round half-to-even behavior
   - `min_pixels` / `max_pixels` handling
   - Qwen3-VL defaults: `factor=32`, `min_pixels=65536`, `max_pixels=16777216`

   Purpose: fix cases like `720x324`, where HF resizes to `704x320` but C++ `std::round` produced `736x320`.

   Result: MNN prompt token count changed from `703` to `693`, matching HF/AutoProcessor for the tested prompt.

3. Export config fields

   Added export-side config fields in `transformers/llm/export/utils/vision.py`:

   ```python
   image_size_unit
   image_min_pixels
   image_max_pixels
   ```

   Purpose: let runtime read resize parameters from model config instead of relying only on hardcoded defaults.

4. Debug dump hooks

   Added environment-gated dumps in `llm.cpp` / `omni.cpp`:

   - `MNN_DUMP_HANDOFF`
   - `MNN_DUMP_HANDOFF_DIR`
   - `MNN_DUMP_VISION_INPUT_DIR`

   Purpose: compare intermediate tensors such as vision inputs/outputs, hidden states, position ids, deepstack inputs, and logits.

   Default behavior is unchanged when these env vars are not set.

5. Temporary exporter no-fuse switches

   Added environment switches in `llmexport.py`:

   ```bash
   MNN_DISABLE_LLM_FUSED_ATTENTION=1
   MNN_DISABLE_LLM_TRANSFORMER_FUSE=1
   ```

   Purpose: try exporting a no-fuse model to test whether fused attention / transformer fuse caused output divergence.

   Result: the no-fuse model export generated files, but runtime failed with a reshape error:

   ```text
   Reshape error: 2048 -> 1024
   Compute Shape Error for /blocks.0/self_attn/Reshape_1_output_0
   ```

   So the current non-fused export path is not directly usable for this Qwen3-VL model.

6. Qwen3-VL MRoPE decode position id fix

   Fixed `Omni::gen_position_ids()` in `transformers/llm/engine/src/omni.cpp`.

   Before the fix, multimodal decode used token length based positions:

   ```cpp
   auto pos = mContext->all_seq_len + i;
   ```

   For the Taobao prompt, prefill MRoPE positions ended at `482`, but decode used `693`.
   That made the second token behave like a context-broken decode and MNN stopped after `{"`.

   After the fix, decode continues from the MRoPE sequence position:

   ```cpp
   auto pos = mContext->gen_seq_len + mPositionIds.back() + i;
   ```

   Verified after rebuild:

   ```text
   call_7 position max = 482
   call_8 position_ids = [483, 483, 483]
   call_8 top1 = 19895 ("reason")
   ```

   MNN then generated a complete JSON response for the same greedy prompt.

## Chat Template Finding

No chat template algorithm was patched.

Findings:

- If passing a fully rendered prompt such as `<|im_start|>system...<|im_start|>assistant` to MNN, config must set:

  ```json
  "use_template": false
  ```

  Otherwise MNN wraps the already-rendered prompt again.

- The earlier `692` vs `693` token count difference came from prompt string mismatch:
  - Torch raw script used `.strip()`, removing the trailing newline.
  - MNN `llm_demo` preserves the prompt file's trailing newline.

  When comparing the exact same string, including the trailing newline, Torch and MNN text token ids are exactly equal:

  ```json
  {
    "torch_len": 693,
    "mnn_len": 693,
    "same_len": true,
    "exact_equal": true
  }
  ```

## Current Status

- Resize interpolation and smart resize alignment are fixed for the tested Qwen3-VL prompt.
- Visual tensor shapes are aligned.
- Text token ids are exactly equal when using the same input string.
- Final generation is now structurally aligned for the tested greedy Taobao prompt: MNN generates a complete JSON response.
- Runtime flash attention was tested with `attention_mode=0`; it did not fix final output.
- The root cause of the `{"` early stop was the Qwen3-VL MRoPE decode position id mismatch, not image token count or chat template wrapping.
