# Qwen3-VL PR Verification Guide

Date: 2026-07-06

This note is the short PR verification record for:

```text
precision_cmp/patches/0001-clean-qwen3-vl-runtime-fixes.patch
```

## Scope

Production files:

```text
transformers/llm/engine/src/omni.cpp
transformers/llm/export/utils/vision.py
```

Patch content:

1. Qwen3-VL `smart_resize` alignment with HF/AutoProcessor.
2. Qwen3-VL Pillow-like BICUBIC image resize path.
3. Qwen3-VL decode-stage MRoPE position fix.
4. Export `image_size_unit`, `image_min_pixels`, `image_max_pixels`.

Do not include debug dumps, private data, or temporary exporter fuse switches in
the production PR.

## Key Requirements

- Use the `mnn` conda environment for Python commands.
- Build `llm_demo` with `-DMNN_BUILD_LLM_OMNI=ON`; a plain LLM build does not
  exercise image preprocessing and can show `vision time = 0.00 s`.
- Treat Torch/HF greedy as a semantic reference, not a token-level equality
  oracle.
- Use tensor cosine comparisons for strict preprocessing/vision alignment.

## Setup

Download the public model:

```bash
mkdir -p /tmp/models/Qwen3-VL-2B-Instruct
conda run -n mnn modelscope download \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --local_dir /tmp/models/Qwen3-VL-2B-Instruct
```

Prepare a public image and resize it to a non-32-aligned source size:

```bash
curl -L \
  "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.jpg" \
  -o /tmp/qwen3vl_public_source.jpg

conda run -n mnn python - <<'PY'
from PIL import Image

img = Image.open("/tmp/qwen3vl_public_source.jpg").convert("RGB")
img = img.resize((720, 324), Image.Resampling.BICUBIC)
img.save("/tmp/qwen3vl_review_720x324.jpg")
print(img.size)
PY
```

Create a one-line MNN prompt. `llm_demo` reads prompt files line by line:

```bash
cat >/tmp/qwen3vl_cmp_prompt.txt <<'EOF'
<img>/tmp/qwen3vl_review_720x324.jpg</img> Describe the image in one short sentence.
EOF
```

Prepare before/after worktrees:

```bash
git fetch origin master
git worktree add -b review/qwen3-vl-runtime-baseline /tmp/mnn-qwen3vl-baseline origin/master
git worktree add -b review/qwen3-vl-runtime-fixes-v2 /tmp/mnn-qwen3vl-review-v2 origin/master

cd /tmp/mnn-qwen3vl-review-v2
git apply /home/yydh/MNN/precision_cmp/patches/0001-clean-qwen3-vl-runtime-fixes.patch
```

Local verification used:

```text
origin/master = 0bff03cbef43c783f44e41484b9f8a0b28bd758d
```

## Build

Build both worktrees with omni:

```bash
cmake -S /tmp/mnn-qwen3vl-baseline \
  -B /tmp/mnn-qwen3vl-baseline-omni-build \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_CONVERTER=OFF
cmake --build /tmp/mnn-qwen3vl-baseline-omni-build --target llm_demo -j2

cmake -S /tmp/mnn-qwen3vl-review-v2 \
  -B /tmp/mnn-qwen3vl-review-v2-omni-build \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_CONVERTER=OFF
cmake --build /tmp/mnn-qwen3vl-review-v2-omni-build --target llm_demo -j2
```

Confirm vision support:

```bash
rg -n "LLM_SUPPORT_VISION|MNN_IMGCODECS" \
  /tmp/mnn-qwen3vl-baseline-omni-build/CMakeFiles/llm.dir/flags.make \
  /tmp/mnn-qwen3vl-review-v2-omni-build/CMakeFiles/llm.dir/flags.make
```

## Export

Export fp16 MNN models before and after:

```bash
cd /tmp/mnn-qwen3vl-baseline
conda run -n mnn python transformers/llm/export/llmexport.py \
  --path /tmp/models/Qwen3-VL-2B-Instruct \
  --export mnn \
  --quant_bit 16 \
  --lm_quant_bit 16 \
  --visual_quant_bit 16 \
  --dst_path /tmp/qwen3vl_mnn_before_cmp1

cd /tmp/mnn-qwen3vl-review-v2
conda run -n mnn python transformers/llm/export/llmexport.py \
  --path /tmp/models/Qwen3-VL-2B-Instruct \
  --export mnn \
  --quant_bit 16 \
  --lm_quant_bit 16 \
  --visual_quant_bit 16 \
  --dst_path /tmp/qwen3vl_mnn_after_cmp1
```

Expected exported config:

```text
before: image_size_unit=None, image_min_pixels=None, image_max_pixels=None
after:  image_size_unit=32, image_min_pixels=65536, image_max_pixels=16777216
```

Set both runtime configs to greedy:

```bash
conda run -n mnn python - <<'PY'
import json

for p in [
    "/tmp/qwen3vl_mnn_before_cmp1/config.json",
    "/tmp/qwen3vl_mnn_after_cmp1/config.json",
]:
    c = json.load(open(p))
    c["sampler_type"] = "greedy"
    c["temperature"] = 1.0
    json.dump(c, open(p, "w"), ensure_ascii=False, indent=4)
PY
```

## Greedy Output Check

Torch/HF greedy reference:

```bash
conda run -n mnn python - <<'PY'
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "/tmp/models/Qwen3-VL-2B-Instruct"
image = Image.open("/tmp/qwen3vl_review_720x324.jpg").convert("RGB")
prompt = "Describe the image in one short sentence."

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval()

messages = [{
    "role": "user",
    "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs, do_sample=False, max_new_tokens=32)

new_tokens = out[:, inputs["input_ids"].shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
PY
```

Observed Torch/HF output:

```text
An elderly woman in a navy uniform stands in front of an American flag.
```

MNN before/after:

```bash
/tmp/mnn-qwen3vl-baseline-omni-build/llm_demo \
  /tmp/qwen3vl_mnn_before_cmp1/config.json \
  /tmp/qwen3vl_cmp_prompt.txt \
  64 \
  nothink

/tmp/mnn-qwen3vl-review-v2-omni-build/llm_demo \
  /tmp/qwen3vl_mnn_after_cmp1/config.json \
  /tmp/qwen3vl_cmp_prompt.txt \
  64 \
  nothink
```

Observed MNN output:

```text
before: AnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAnAn...
after:  An elderly man in a U.S. Navy uniform stands in front of an American flag.
```

Both MNN runs exercised vision:

```text
before: prompt tokens=187, vision time=4.42 s, pixels_mp=0.17 MP
after:  prompt tokens=187, vision time=7.72 s, pixels_mp=0.17 MP
```

Interpretation:

- Patch after is semantically aligned with Torch/HF greedy.
- Patch before shows repeated-token failure.
- Exact text equality is not required for this smoke test.

## Numerical Evidence

Existing tensor comparison result:

| Tensor | Before | After |
| --- | ---: | ---: |
| image patches / pixel values | 0.9920087 | 0.9999987 |
| vision encoder `image_embeds` | 0.9024969 | 0.9999431 |
| `deepstack_feature` | 0.9583789 | 0.9999789 |

Use this as the stronger evidence for preprocessing/vision feature alignment.

## Review Notes

- Run `clang-format -i transformers/llm/engine/src/omni.cpp`.
- Run `python -m py_compile transformers/llm/export/utils/vision.py`.
- Run `git diff --check`.
- The Pillow-like helper is currently local to `omni.cpp`. It can be split into
  an LLM-internal vision preprocess utility if maintainability is prioritized,
  but should not become a generic CV API without public semantics and tests.
