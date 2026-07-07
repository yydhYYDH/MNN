# Qwen3-VL MNN Output Alignment Fix Report

Date: 2026-07-06

## Background

The UI-Venus Qwen3-VL model produced valid output before conversion in Torch/vLLM, but after exporting to MNN the runtime output was abnormal.

For the Taobao screenshot prompt, MNN originally stopped almost immediately:

```text
{"
```

The expected behavior was a complete JSON action response similar to the Torch/vLLM output.

The tested prompt and image were:

```text
/home/ma-user/workspace/csm/mobiinfer/precision_cmp/outputs/vllm_style_prompt/prompt_vllm_style_mnn.txt
/home/ma-user/workspace/csm/mobi-autoround/data/taobao/mnn_test.jpg
```

The runtime config used greedy decoding and disabled template wrapping:

```text
/home/ma-user/workspace/csm/mobiinfer/precision_cmp/outputs/full_vlm/config.greedy.no_template.no_flash.json
```

## Fix Summary

Four classes of issues were investigated and fixed or ruled out.

## 1. Image Resize Alignment

MNN runtime preprocessing did not initially match HF/AutoProcessor.

Two differences were fixed in `transformers/llm/engine/src/omni.cpp`:

1. Qwen3-VL `smart_resize` logic was aligned with HF:
   - Python round half-to-even behavior
   - `min_pixels`
   - `max_pixels`
   - size factor `patch_size * merge_size = 32`
2. Qwen3-VL image resize interpolation was changed from MNN linear resize to a Pillow-like BICUBIC path.

This mattered for the tested image size:

```text
Input size: 324x720
HF smart_resize result: 320x704
Old C++ rounding could produce: 320x736
```

After this fix, visual grid and image placeholder count aligned with HF.

### Pillow BICUBIC Implementation

HF/AutoProcessor uses Pillow-style BICUBIC image resize. MNN previously used:

```cpp
MNN::CV::resize(..., MNN::CV::INTER_LINEAR, ...)
```

for Qwen3-VL image preprocessing.

The runtime now has a Qwen3-VL specific Pillow-like resize helper:

```cpp
qwen3PillowLikeResize(image, mVisionWidth, mVisionHeight)
```

It implements:

```text
Catmull-Rom style Pillow bicubic kernel
separable horizontal + vertical resize
Pillow-like coefficient normalization
fixed-point coefficient quantization
BGR to RGB conversion
Qwen3-VL normalization to [-1, 1]
```

The path is enabled by default only for Qwen3-VL:

```cpp
bool usePillowResize = isQwen3VL && mConfig->config_.value("qwen3_vl_pillow_resize", true);
```

It can be disabled from config if needed:

```json
{
  "qwen3_vl_pillow_resize": false
}
```

Current status:

```text
smart_resize size alignment: applied
Pillow-like BICUBIC interpolation: applied for Qwen3-VL
```

## 2. Prompt And Chat Template Alignment

The full prompt was already rendered in vLLM/HF style:

```text
<|im_start|>system
...
<|im_start|>assistant
```

Therefore MNN must not apply another chat template. The config must use:

```json
{
  "use_template": false
}
```

One earlier token count mismatch was caused by string handling:

```text
Torch script used .strip(), removing the final newline.
MNN llm_demo preserved the prompt file newline.
```

After comparing the exact same string, including the final newline:

```json
{
  "torch_len": 693,
  "mnn_len": 693,
  "exact_equal": true
}
```

So text token ids were confirmed identical.

## 3. Handoff Dump And Input Verification

Debug dumps were added to compare the handoff between vision preprocessing and LLM forward.

Environment switches:

```bash
MNN_DUMP_VISION_INPUT_DIR=...
MNN_DUMP_HANDOFF=1
MNN_DUMP_HANDOFF_DIR=...
MNN_DUMP_HANDOFF_LIMIT=...
```

Compared tensors included:

```text
patches
image_embeds
deepstack_feature
hidden_state
position_ids
deepstack_embeds
logits
```

Key findings:

```text
position_ids: exact match at prefill
hidden_state cosine: 0.999921
deepstack_embeds cosine: 0.999967
prefill step0 top1: {" on both sides
```

KV cache was also checked. MNN decode used historical KV:

```text
prefill: kvSeqLen = 693
decode:  kvSeqLen = 694
```

So the issue was not caused by missing KV cache.

## 4. Root Cause: Qwen3-VL MRoPE Decode Position

The real root cause was in `Omni::gen_position_ids()` for multimodal MRoPE decode.

Old code:

```cpp
auto pos = mContext->all_seq_len + i;
```

This is wrong for Qwen3-VL multimodal MRoPE.

`mContext->all_seq_len` is token count. But Qwen3-VL MRoPE position is not equal to token index when images are present. Image placeholder tokens are many, while MRoPE positions advance according to the visual `t/h/w` grid and then continue with text positions.

For the tested prompt:

```text
prompt token count = 693
prefill MRoPE max position = 482
```

The next decode token should therefore use position:

```text
483
```

But old MNN code used:

```text
693
```

This caused the second generated token to become `<|im_end|>`, so output stopped after:

```text
{"
```

The fix:

```cpp
auto pos = mContext->gen_seq_len + mPositionIds.back() + i;
```

After the fix:

```text
call_7 position max = 482
call_8 position_ids = [483, 483, 483]
call_8 top1 = 19895 ("reason")
```

MNN then generated a complete JSON response.

## Verification Result

After rebuilding:

```bash
cmake --build build --target llm_demo -j$(nproc)
```

The MNN output became:

```json
{
  "reasoning": "当前位于淘宝首页，页面顶部有一个搜索框，显示“静音风扇”字样。任务目标是购买雨伞，因此我需要通过搜索功能查找雨伞。我将点击搜索框并输入“雨伞”进行搜索。",
  "action": "click_input",
  "parameters": {
    "target_element": "顶部的搜索框，当前显示“静音风扇”",
    "text": "雨伞",
    "bbox": [164, 71, 764, 111]
  }
}
```

This is structurally valid and semantically aligned with the Torch/vLLM result.

The output is not guaranteed to be byte-identical to HF/vLLM because minor numeric differences remain in image preprocessing and MNN runtime computation, but the previous hard failure was fixed.

## Impact Scope

This fix should not affect ordinary text-only LLMs:

```text
Text-only models use Llm::gen_position_ids().
```

It should not affect non-MRoPE multimodal models:

```text
If position_ids first dimension is 1, Omni::gen_position_ids() falls back to Llm::gen_position_ids().
```

It affects multimodal MRoPE models such as Qwen-VL style models:

```text
Qwen2-VL / Qwen2.5-VL / Qwen3-VL style MRoPE decode
```

For these models, the fix is expected to be correct because decode positions must continue from the last prefill MRoPE position, not from raw token count.

Pure text prompts running through Omni MRoPE should remain effectively unchanged, because without visual compression:

```text
mPositionIds.back() == prompt_len - 1
```

So the new decode position matches the old token-index-based position.

## Files Changed

Main runtime fix:

```text
transformers/llm/engine/src/omni.cpp
```

Diagnostic scripts:

```text
precision_cmp/scripts/compare_export_mnn_handoff_exact.py
precision_cmp/scripts/inspect_mnn_handoff_dump.py
```

Related summary:

```text
precision_cmp/notes/qwen3_vl_mnn_alignment_summary.md
```
