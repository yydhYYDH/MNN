# Qwen3-VL MNN Patch Set

## Review This Patch

```text
0001-clean-qwen3-vl-runtime-fixes.patch
```

It contains the production fix:

1. Qwen3-VL `smart_resize` alignment.
2. Qwen3-VL Pillow-like BICUBIC resize path.
3. Qwen3-VL decode MRoPE position fix.
4. Export config fields:
   - `image_size_unit`
   - `image_min_pixels`
   - `image_max_pixels`

## Verification

Use:

```text
../notes/qwen3_vl_pr_verification_guide.md
```

Required build flag:

```text
-DMNN_BUILD_LLM_OMNI=ON
```

Without omni, the image path is not exercised and `vision time` can be `0.00 s`.

Observed smoke result:

```text
before: repeated-token failure, e.g. AnAnAn...
after:  coherent image-grounded output, semantically aligned with Torch/HF greedy
```

Strong numerical evidence:

| Tensor | Before | After |
| --- | ---: | ---: |
| image patches / pixel values | 0.9920087 | 0.9999987 |
| vision encoder `image_embeds` | 0.9024969 | 0.9999431 |
| `deepstack_feature` | 0.9583789 | 0.9999789 |

## Other Patches

These are investigation/debug patches, not production PR content:

```text
0000-all-mnn-qwen3-vl-debug-fixes.patch
0001-qwen3-vl-runtime-resize-mrope.patch
0002-llm-handoff-debug-dumps.patch
0003-exporter-fuse-env-switches.patch
```
