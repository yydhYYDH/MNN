#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import torch

ROOT = Path("/home/ma-user/workspace/csm/mobiinfer")
EXPORT_PY = ROOT / "transformers/llm/export"
DEFAULT_MODEL_DIR = Path(
    "/home/ma-user/modelarts/user-job-dir/lzz/models/"
    "UI-Venus-1.5-2B-0422-reasoning-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step100"
)
DEFAULT_EXPORT_DIR = ROOT / "transformers/llm/export/model"
DEFAULT_PROMPT = ROOT / "precision_cmp/outputs/vllm_style_prompt/prompt_vllm_style_mnn.txt"
DEFAULT_MNN = ROOT / "precision_cmp/outputs/official_vs_mnn_handoff/mnn_handoff"
DEFAULT_OUT = ROOT / "precision_cmp/outputs/official_vs_mnn_handoff/export_mnn_exact_handoff.json"

if importlib.util.find_spec("onnx") is None and "onnx" not in sys.modules:
    fake_onnx = types.ModuleType("onnx")
    fake_onnx.save = lambda *args, **kwargs: None
    fake_onnx.load = lambda *args, **kwargs: None
    sys.modules["onnx"] = fake_onnx

sys.path.insert(0, str(EXPORT_PY))

from llmexport import LlmExporter, build_args  # noqa: E402


def to_np(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return value.astype(dtype) if dtype is not None else value


def load_mnn_dump(root: Path, name: str):
    meta = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    dtype = {"float32": np.float32, "float": np.float32, "int32": np.int32, "int64": np.int64}[meta["dtype"]]
    arr = np.fromfile(root / f"{name}.raw", dtype=dtype).reshape(meta["shape"])
    return arr


def topk(logits, k=10):
    flat = np.asarray(logits).reshape(-1)
    idx = np.argpartition(-flat, min(k, flat.size - 1))[:k]
    idx = idx[np.argsort(-flat[idx])]
    return idx.astype(np.int64), flat[idx].astype(np.float32)


def numeric_cmp(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    report = {"shape_a": list(a.shape), "shape_b": list(b.shape), "same_shape": a.shape == b.shape}
    if a.shape != b.shape:
        return report
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(b.dtype, np.integer):
        neq = a != b
        report.update(
            {
                "exact_equal": bool(not np.any(neq)),
                "num_diff": int(np.count_nonzero(neq)),
                "max_abs": int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if a.size else 0,
            }
        )
        if np.any(neq):
            idx = np.argwhere(neq.reshape(-1))[:20].reshape(-1)
            af = a.reshape(-1)
            bf = b.reshape(-1)
            report["first_diffs"] = [
                {"flat_index": int(i), "a": int(af[i]), "b": int(bf[i])} for i in idx
            ]
        return report
    af = a.astype(np.float64).reshape(-1)
    bf = b.astype(np.float64).reshape(-1)
    diff = af - bf
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    report.update(
        {
            "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
            "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
            "cosine": float(np.dot(af, bf) / denom) if denom != 0 else None,
            "a_min": float(np.min(af)) if af.size else None,
            "a_max": float(np.max(af)) if af.size else None,
            "b_min": float(np.min(bf)) if bf.size else None,
            "b_max": float(np.max(bf)) if bf.size else None,
        }
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--mnn-dump", type=Path, default=DEFAULT_MNN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    arg_parser = argparse.ArgumentParser()
    build_args(arg_parser)
    parsed = arg_parser.parse_args(["--path", str(args.model_dir)])
    parsed.path = str(args.model_dir)
    parsed.tokenizer_path = str(args.model_dir)
    parsed.dst_path = str(args.export_dir)
    parsed.export = None
    parsed.skip_weight = False

    prompt = args.prompt.read_text(encoding="utf-8")
    with torch.no_grad():
        exporter = LlmExporter(parsed)
        exporter.model.eval()
        model = exporter.model
        visual = model.visual
        tokenizer = exporter.tokenizer
        input_ids = visual.str_to_ids(prompt) if visual is not None else exporter.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        seq_len = int(input_ids.numel())
        attention_mask = model.get_attention_mask(seq_len, 0)
        position_ids = model.get_position_ids(seq_len, 0, input_ids)
        input_embeds = model.embedding(input_ids)
        deepstack_embeds = visual.deepstacks() if visual is not None else None
        logits0, _, _ = model.forward(
            input_ids=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logits_index=torch.tensor([-1], dtype=torch.int32),
            deepstack_embeds=deepstack_embeds,
        )
        logits0 = logits0[:, -1, :].float()
        next_id = torch.argmax(logits0, dim=-1).to(torch.int64)

        decode_seq_len = seq_len + 1
        decode_attention_mask = model.get_attention_mask(decode_seq_len, 1)
        decode_position_ids = model.get_position_ids(decode_seq_len, 1, next_id)
        decode_embeds = model.embedding(next_id)
        decode_deepstack = visual.deepstacks() if visual is not None else None
        logits1, _, _ = model.forward(
            input_ids=decode_embeds,
            attention_mask=decode_attention_mask,
            position_ids=decode_position_ids,
            logits_index=torch.tensor([-1], dtype=torch.int32),
            deepstack_embeds=decode_deepstack,
        )
        logits1 = logits1[:, -1, :].float()

    torch_arrays = {
        "input_ids": to_np(input_ids, np.int64).reshape(-1),
        "attention_mask": to_np(attention_mask, np.float32),
        "position_ids": to_np(position_ids, np.int32),
        "hidden_state": to_np(input_embeds, np.float32),
    }
    if deepstack_embeds is not None:
        torch_arrays["extra_0"] = to_np(deepstack_embeds, np.float32)

    mnn_arrays = {
        "attention_mask": load_mnn_dump(args.mnn_dump, "call_7_attention_mask"),
        "position_ids": load_mnn_dump(args.mnn_dump, "call_7_position_ids"),
        "hidden_state": load_mnn_dump(args.mnn_dump, "call_7_hidden_state"),
        "extra_0": load_mnn_dump(args.mnn_dump, "call_7_extra_0"),
        "logits0": load_mnn_dump(args.mnn_dump, "call_7_logits"),
        "logits1": load_mnn_dump(args.mnn_dump, "call_8_logits"),
    }
    torch_arrays["logits0"] = to_np(logits0, np.float32).reshape(-1)
    torch_arrays["logits1"] = to_np(logits1, np.float32).reshape(-1)

    report = {
        "prompt": str(args.prompt),
        "prompt_len_chars": len(prompt),
        "seq_len": seq_len,
        "torch_input_ids_head": torch_arrays["input_ids"][:32].tolist(),
        "torch_input_ids_tail": torch_arrays["input_ids"][-32:].tolist(),
        "image_pad_count": int(np.count_nonzero(torch_arrays["input_ids"] == getattr(visual, "image_pad_id", -1))),
        "export_step0_token": int(next_id.reshape(-1)[0]),
        "export_step0_text": tokenizer.id_to_str(int(next_id.reshape(-1)[0])),
        "comparisons": {},
        "topk": {},
    }
    for key in ["attention_mask", "position_ids", "hidden_state", "extra_0", "logits0", "logits1"]:
        if key in torch_arrays and key in mnn_arrays:
            report["comparisons"][key] = numeric_cmp(torch_arrays[key], mnn_arrays[key])
    for key in ["logits0", "logits1"]:
        if key in torch_arrays and key in mnn_arrays:
            torch_ids, torch_vals = topk(torch_arrays[key])
            mnn_ids, mnn_vals = topk(mnn_arrays[key])
            report["topk"][key] = {
                "torch_ids": torch_ids.tolist(),
                "torch_vals": torch_vals.tolist(),
                "torch_text": [tokenizer.id_to_str(int(i)) for i in torch_ids],
                "mnn_ids": mnn_ids.tolist(),
                "mnn_vals": mnn_vals.tolist(),
                "mnn_text": [tokenizer.id_to_str(int(i)) for i in mnn_ids],
                "same_top1": bool(torch_ids[0] == mnn_ids[0]),
                "top10_overlap": int(len(set(torch_ids.tolist()) & set(mnn_ids.tolist()))),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
