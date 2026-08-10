#!/usr/bin/env python3

import argparse
import array
import math
import sys


def load_binary(path):
    values = array.array("f")
    with open(path, "rb") as stream:
        values.frombytes(stream.read())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def load_text(path):
    with open(path, "r", encoding="utf-8") as stream:
        return array.array("f", (float(line) for line in stream if line.strip()))


def main():
    parser = argparse.ArgumentParser(description="Compare float32 MNN reference and output tensors")
    parser.add_argument("reference", help="little-endian float32 reference file")
    parser.add_argument("actual", help="little-endian float32 or one-value-per-line output file")
    parser.add_argument("--actual-text", action="store_true", help="read the actual output as text")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-nrmse", type=float, default=0.04)
    parser.add_argument("--max-abs", type=float, default=math.inf)
    args = parser.parse_args()

    reference = load_binary(args.reference)
    actual = load_text(args.actual) if args.actual_text else load_binary(args.actual)
    if not reference or len(reference) != len(actual):
        print(f"size mismatch: reference={len(reference)} actual={len(actual)}", file=sys.stderr)
        return 2

    squared_error = 0.0
    squared_reference = 0.0
    squared_actual = 0.0
    dot = 0.0
    max_error = 0.0
    max_index = 0
    non_finite = 0
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        if not math.isfinite(expected) or not math.isfinite(observed):
            non_finite += 1
            continue
        error = abs(observed - expected)
        if error > max_error:
            max_error = error
            max_index = index
        squared_error += error * error
        squared_reference += expected * expected
        squared_actual += observed * observed
        dot += expected * observed

    count = len(reference)
    rms = math.sqrt(squared_error / count)
    reference_rms = math.sqrt(squared_reference / count)
    nrmse = rms / reference_rms if reference_rms else math.inf
    cosine = dot / math.sqrt(squared_reference * squared_actual) if squared_reference and squared_actual else 0.0
    print(
        f"elements={count} non_finite={non_finite} max_error={max_error:.9g} max_index={max_index} "
        f"rms_error={rms:.9g} nrmse={nrmse:.9g} cosine={cosine:.12g}"
    )
    return 0 if non_finite == 0 and max_error <= args.max_abs and nrmse <= args.max_nrmse and cosine >= args.min_cosine else 1


if __name__ == "__main__":
    sys.exit(main())
