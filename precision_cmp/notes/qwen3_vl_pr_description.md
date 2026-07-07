## Description

Fix Qwen3-VL multimodal output mismatch after exporting to MNN.

This PR aligns the Qwen3-VL runtime/export path with HuggingFace/AutoProcessor:

1. Align Qwen3-VL `smart_resize`
   - Uses `image_size_unit = patch_size * merge_size`
   - Uses exported `image_min_pixels` and `image_max_pixels`
   - Matches Python round-half-to-even behavior

2. Add a Qwen3-VL Pillow-like BICUBIC resize path
   - HF/AutoProcessor uses Pillow-style BICUBIC image resizing
   - The previous MNN runtime path used linear resize here
   - The new path is scoped to Qwen3-VL

3. Fix Qwen3-VL decode-stage MRoPE position
   - Decode continues from the last prefill MRoPE position
   - It no longer uses raw accumulated token length for multimodal MRoPE decode

4. Export Qwen3-VL resize config
   - `image_size_unit`
   - `image_min_pixels`
   - `image_max_pixels`

Verification summary:

```text
Model: Qwen/Qwen3-VL-2B-Instruct
Build: -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON -DMNN_LOW_MEMORY=ON
Export: --quant_bit 16 --lm_quant_bit 16 --visual_quant_bit 16
```

Exported config after this patch:

```text
image_size_unit=32
image_min_pixels=65536
image_max_pixels=16777216
```

End-to-end greedy smoke test with the same image and prompt:

```text
before: repeated-token failure, e.g. AnAnAn...
after:  coherent image-grounded output, semantically aligned with Torch/HF greedy
```

Tensor comparison against HF/AutoProcessor:

| Tensor | Before | After |
| --- | ---: | ---: |
| image patches / pixel values | 0.9920087 | 0.9999987 |
| vision encoder `image_embeds` | 0.9024969 | 0.9999431 |
| `deepstack_feature` | 0.9583789 | 0.9999789 |

Runtime verification command shape:

```bash
cmake -S . -B build-qwen3vl \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_CONVERTER=OFF

cmake --build build-qwen3vl --target llm_demo -j2

./build-qwen3vl/llm_demo /path/to/exported/config.json /path/to/prompt.txt 64 nothink
```

Note: `llm_demo` must be built with `MNN_BUILD_LLM_OMNI=ON`; otherwise the image path is not exercised and `vision time` can be `0.00 s`.

## Module

LLM

## Type

- [ ] Feature
- [x] Bugfix
- [ ] Perf
- [ ] Refact
- [ ] Style
- [ ] Doc
- [ ] Test
- [ ] Chore

## Checklist

- [x] Commit message follows `[Module:Type] Description` format
- [x] Code compiles without errors
- [x] Tested on relevant platform(s)
- [x] No unrelated format or style changes included
