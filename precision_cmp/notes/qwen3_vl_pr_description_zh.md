## Description

修复 Qwen3-VL 导出到 MNN 后的多模态输出异常问题。

本 PR 将 MNN Qwen3-VL 的 runtime/export 路径与 HuggingFace/AutoProcessor进行对齐：

1. 对齐huggingface中 Qwen3-VL 的 `smart_resize`
   - 使用 `image_size_unit = patch_size * merge_size`
   - 使用导出的 `image_min_pixels` 和 `image_max_pixels`
   - 对齐 Python round-half-to-even 行为

2. 增加huggingface中 Qwen3-VL resize用的 Pillow库 的 BICUBIC resize 
   - HF/AutoProcessor 使用 Pillow 风格的 BICUBIC 图像缩放
   - 原 MNN runtime 这里使用的是 linear resize
   - 新路径仅作用于 Qwen3-VL

3. 修复 Qwen3-VL decode 阶段 MRoPE position
   - decode 阶段从 prefill 最后的 MRoPE position 继续
   - 不再使用多模态场景下不正确的原始累计 token length 
 
从我的测试结果来看，第3点对精度影响最大。

### 疑惑点
 对于第3点，我有些疑惑 ，根据测试，源代码中被注释掉的那行代码反而是正确的写法：
 源代码在transformers/llm/engine/src/omni.cpp 1452行的位置
 ```
            // auto pos = mContext->gen_seq_len + mPositionIds.back() + i;  // 正确的写法，但是被注释了
            auto pos = mContext->all_seq_len + i;
```
修复后，应为
 ```
            auto pos = mContext->gen_seq_len + mPositionIds.back() + i;  
```

### 验证结果

#### 端到端验证
修改后模型输出正常。用Qwen3-VL-2B-Instrcut，输入相同的图片和prompt：

- 修改前:  `AnAnAnAnAnAn...` 重复 token 异常
- 修改后:  `An elderly man in a U.S. Navy uniform stands in front of an American flag.`  可理解的图像描述输出，语义上与 Torch/HF greedy 对齐

#### vision部分验证
修改后与 HF/AutoProcessor 的 tensor 的余弦相似度提升到了0.999，对比如下：

| Tensor | Before | After |
| --- | ---: | ---: |
| image patches / pixel values | 0.9920087 | 0.9999987 |
| vision encoder `image_embeds` | 0.9024969 | 0.9999431 |
| `deepstack_feature` | 0.9583789 | 0.9999789 |

### 验证步骤

#### 端到端验证
准备公开图片和 prompt：

```bash
curl -L \
  "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.jpg" \
  -o /tmp/qwen3vl_public_source.jpg

python - <<'PY'
from PIL import Image

img = Image.open("/tmp/qwen3vl_public_source.jpg").convert("RGB")
img = img.resize((720, 324), Image.Resampling.BICUBIC)
img.save("/tmp/qwen3vl_review_720x324.jpg")
print(img.size)
PY

cat >/tmp/qwen3vl_prompt.txt <<'EOF'
<img>/tmp/qwen3vl_review_720x324.jpg</img> Describe the image in one short sentence.
EOF
```

构建 MNN runtime：

```bash
cmake -S . -B build-qwen3vl \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_LOW_MEMORY=ON 

cmake --build build-qwen3vl --target llm_demo -j16

```

导出 fp16 MNN 模型：

```bash
conda run -n mnn python transformers/llm/export/llmexport.py \
  --path /tmp/models/Qwen3-VL-2B-Instruct \
  --export mnn \
  --quant_bit 16 \
  --lm_quant_bit 16 \
  --visual_quant_bit 16 \
  --dst_path /tmp/qwen3vl_mnn
```

检查 patch 后导出的配置：

```bash
conda run -n mnn python - <<'PY'
import json

c = json.load(open("/tmp/qwen3vl_mnn/llm_config.json"))
print("image_size_unit =", c.get("image_size_unit"))
print("image_min_pixels =", c.get("image_min_pixels"))
print("image_max_pixels =", c.get("image_max_pixels"))
PY
```

期望输出：

```text
image_size_unit=32
image_min_pixels=65536
image_max_pixels=16777216
```

设置 greedy 采样：

```bash
conda run -n mnn python - <<'PY'
import json

p = "/tmp/qwen3vl_mnn/config.json"
c = json.load(open(p))
c["sampler_type"] = "greedy"
c["temperature"] = 1.0
json.dump(c, open(p, "w"), ensure_ascii=False, indent=4)
PY
```

Torch/HF greedy 参考输出：

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

MNN greedy 输出验证：

```bash
./build-qwen3vl/llm_demo /tmp/qwen3vl_mnn/config.json /tmp/qwen3vl_prompt.txt 64 
```

同图同 prompt 的端到端 greedy smoke test：

```text
before: 重复 token 异常，例如 AnAnAn...
after:  可理解的图像描述输出，语义上与 Torch/HF greedy 对齐
```

#### vision部分验证
与 HF/AutoProcessor 的 tensor 对比：
验证方式是使用同一张图片和同一份 Qwen3-VL processor 配置：

1. 用 HF `AutoProcessor` 导出参考 tensor：
   - `pixel_values`
   - `image_grid_thw`
2. 在 MNN runtime 中临时 dump 对应的中间结果：
   - image preprocessing 后、进入 vision encoder 前的 image patches
   - vision encoder 输出 `image_embeds`
   - Qwen3-VL 使用的 `deepstack_feature`
3. 将 HF 和 MNN 的 tensor reshape 到相同形状后计算 cosine similarity。



该对比用于验证 `smart_resize`、Pillow-like BICUBIC resize、归一化、patch layout 以及 vision feature 是否对齐；它比最终文本输出更适合作为数值正确性证据。

临时 dump 代码位置和示例：

1. HF 参考 tensor dump：

```python
import numpy as np
from PIL import Image
from transformers import AutoProcessor

model_path = "/tmp/models/Qwen3-VL-2B-Instruct"
image_path = "/tmp/qwen3vl_review_720x324.jpg"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
image = Image.open(image_path).convert("RGB")

inputs = processor(
    text=[
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the image in one short sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ],
    images=[image],
    return_tensors="pt",
)

np.save("/tmp/hf_pixel_values.npy", inputs["pixel_values"].cpu().float().numpy())
np.save("/tmp/hf_image_grid_thw.npy", inputs["image_grid_thw"].cpu().numpy())
print("pixel_values:", tuple(inputs["pixel_values"].shape))
print("image_grid_thw:", inputs["image_grid_thw"])
```

2. MNN runtime 临时 dump 插桩。

在 `transformers/llm/engine/src/omni.cpp` 中临时加入 helper：

```cpp
static void dumpFloatVar(const std::string& path, MNN::Express::VARP var) {
    auto info = var->getInfo();
    if (info == nullptr) {
        return;
    }
    const float* ptr = var->readMap<float>();
    if (ptr == nullptr) {
        return;
    }

    int size = 1;
    for (auto d : info->dim) {
        size *= d;
    }

    FILE* fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        return;
    }
    fwrite(ptr, sizeof(float), size, fp);
    fclose(fp);
}
```

在 `Omni::qwen2VisionProcess(VARP image)` 中，最终送入 vision encoder 的 image patches 变量就是 `patches`。在最后一次 reshape 之后、构造 `moduleInputs` 之前加入：

```cpp
patches =
    Express::_Reshape(patches, {grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size});
dumpFloatVar("/tmp/mnn_pixel_values.f32", patches);
```

在 vision encoder 输出后，`outputs[0]` 是 `image_embeds`。在 `mVisionModule->onForward(moduleInputs)` 返回后加入：

```cpp
auto outputs = mVisionModule->onForward(moduleInputs);
auto imageEmbedding = outputs[0];
dumpFloatVar("/tmp/mnn_image_embeds.f32", imageEmbedding);
```

当前 Qwen3-VL runtime 会在 `outputs.size() == 2` 时把 `outputs[1]` 存入 `mDeepStackEmbeddings`，因此这里的 `outputs[1]` 就是后续 LLM 使用的 `deepstack_feature`。在现有 push 逻辑前加入：

```cpp
if (outputs.size() == 2) {
    dumpFloatVar("/tmp/mnn_deepstack_feature.f32", outputs[1]);
    mDeepStackEmbeddings.push_back(outputs[1]);
}
```

3. cosine similarity 计算：

```python
import numpy as np

def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

hf_pixel = np.load("/tmp/hf_pixel_values.npy")
mnn_pixel = np.fromfile("/tmp/mnn_pixel_values.f32", dtype=np.float32).reshape(hf_pixel.shape)

print("pixel_values cosine:", cosine(hf_pixel, mnn_pixel))
print("pixel_values max abs diff:", np.max(np.abs(hf_pixel - mnn_pixel)))
print("pixel_values mean abs diff:", np.mean(np.abs(hf_pixel - mnn_pixel)))

# image_embeds / deepstack_feature 同理：
# hf = np.load("/tmp/hf_image_embeds.npy")
# mnn = np.fromfile("/tmp/mnn_image_embeds.f32", dtype=np.float32).reshape(hf.shape)
# print(cosine(hf, mnn))
```


| Tensor | Before | After |
| --- | ---: | ---: |
| image patches / pixel values | 0.9920087 | 0.9999987 |
| vision encoder `image_embeds` | 0.9024969 | 0.9999431 |
| `deepstack_feature` | 0.9583789 | 0.9999789 |




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
