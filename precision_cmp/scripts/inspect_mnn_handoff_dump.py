#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


def load_dump(path: Path):
    meta = json.loads(path.read_text(encoding="utf-8"))
    dtype = {
        "int32": np.int32,
        "int64": np.int64,
        "float32": np.float32,
        "float": np.float32,
    }[meta["dtype"]]
    arr = np.fromfile(path.with_suffix(".raw"), dtype=dtype).reshape(meta["shape"])
    return meta, arr


def main():
    root = Path(
        "/home/ma-user/workspace/csm/mobiinfer/"
        "precision_cmp/outputs/official_vs_mnn_handoff/mnn_handoff"
    )
    names = [
        "call_7_position_ids",
        "call_8_position_ids",
        "call_7_extra_0",
        "call_8_extra_0",
        "call_7_attention_mask",
        "call_8_attention_mask",
        "call_7_hidden_state",
        "call_8_hidden_state",
        "call_7_logits_index",
        "call_8_logits_index",
    ]
    for name in names:
        meta_path = root / f"{name}.json"
        if not meta_path.exists():
            continue
        meta, arr = load_dump(meta_path)
        flat = arr.reshape(-1)
        print(f"\n{name}: shape={arr.shape} dtype={arr.dtype}")
        print(f"  min={flat.min()} max={flat.max()} mean={flat.mean()} sum={flat.sum()}")
        print(f"  head={flat[:24].tolist()}")
        print(f"  tail={flat[-24:].tolist()}")
        if name.endswith("position_ids") and arr.ndim == 2:
            print(f"  rows_head={[arr[i, :24].tolist() for i in range(arr.shape[0])]}")
            print(f"  rows_tail={[arr[i, -24:].tolist() for i in range(arr.shape[0])]}")


if __name__ == "__main__":
    main()
