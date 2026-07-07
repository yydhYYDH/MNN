#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


MODEL = "/home/ma-user/modelarts/user-job-dir/lzz/models/UI-Venus-1.5-2B-0422-reasoning-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step100"
IMAGE = "/home/ma-user/workspace/csm/mobi-autoround/data/taobao/mnn_test.jpg"

SYSTEM_PROMPT = """You are a phone-use AI agent. 

### Action Space
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: open_app, Parameters: app_name (the name of the application to open).
- Name: press_home, Parameters: (no parameters, returns to the home screen).
- Name: press_back, Parameters: (no parameters, goes back to the previous screen).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).

### Response Format
Your output should be a JSON object with the following format:
{  
  "reasoning": "Your reasoning here", 
  "action": "The next action (one of click, click_input, input, swipe, open_app, press_home, press_back, wait, done)", 
  "parameters": {"param1": "value1", "param2": "value2", ...}
}

### Constraints
- If the screen has not changed after your last action, do not repeat the exact same action. Try a different method or slightly adjust coordinates.
- If the task is completed, verify the result before outputting 'done'.
"""

USER_PROMPT = """### Current Task
"去买雨伞"
### Action History
The sequence of actions you have already taken:
(No history)

Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action."""


def scaled_image(path: str, scale: float) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if scale != 1.0:
        size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def as_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def topk(arr, k=10):
    arr = arr.reshape(-1).astype(np.float64)
    idx = np.argpartition(-arr, min(k, arr.size - 1))[:k]
    idx = idx[np.argsort(-arr[idx])]
    return idx.astype(np.int64), arr[idx].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--out-dir", default="/home/ma-user/workspace/csm/mobiinfer/precision_cmp/outputs/official_handoff")
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    image = scaled_image(args.image, args.scale)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": args.image}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image, padding=True, return_tensors="pt")

    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    model.eval()

    captured = {}

    def pre_hook(_module, _module_inputs, kwargs):
        captured["forward_kwargs"] = {k: as_numpy(v) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}

    handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    handle.remove()

    logits = outputs.logits[:, -1, :].float().detach().cpu().numpy().reshape(-1)
    top_ids, top_vals = topk(logits, 10)

    arrays = {
        "input_ids": as_numpy(inputs["input_ids"]).astype(np.int64),
        "first_token_logits": logits.astype(np.float32),
        "top10_ids": top_ids,
        "top10_logits": top_vals,
    }
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            arrays[f"processor_{key}"] = as_numpy(value)
    for key, value in captured.get("forward_kwargs", {}).items():
        arrays[f"forward_{key}"] = value
    np.savez(out / "official_handoff.npz", **arrays)

    report = {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "scaled_size": list(image.size),
        "rendered_prompt_tail": text[-500:],
        "input_keys": list(inputs.keys()),
        "input_shapes": {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)},
        "top10_ids": top_ids.tolist(),
        "top10_logits": top_vals.tolist(),
        "top10_text": [tokenizer.decode([int(i)]) for i in top_ids],
    }
    (out / "official_handoff_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
