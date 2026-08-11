# MNN Hexagon Backend

This directory contains the Hexagon backend implementation for MNN. It supports accelerating model inference on Qualcomm Hexagon DSPs.

## Prerequisites

- **Hexagon SDK**: Ensure you have downloaded and installed the Qualcomm Hexagon SDK.
- **Environment Variable**: Set the `HEXAGON_SDK_ROOT` environment variable pointing to the root of your Hexagon SDK installation.

## Compilation Steps

### 1. Compile the Custom HTP Ops Library

Before compiling MNN, you need to build the custom Hexagon Tensor Processor (HTP) operator library located in `htp-ops-lib/`.

```bash
cd source/backend/hexagon/htp-ops-lib

# Pass the target DSP architecture version (e.g., v73, v75, v79)
# Ensure your Hexagon SDK environment is properly set up before running this script
bash build.sh v79
```

This script will generate two essential libraries in the `htp-ops-lib/outputs/` directory:
- `libMNN_htpops.so`: The Android AArch64 stub library that runs on the CPU.
- `libMNN_htpops_skel.so`: The Hexagon DSP skeleton library that runs on the Hexagon NPU/DSP.

### 2. Compile MNN with Hexagon Backend Enabled

When configuring the MNN build, you need to enable the Hexagon backend by passing the `-DMNN_HEXAGON=ON` flag to CMake. Don't need Hexagon SDK.

```bash
cd /path/to/MNN
mkdir build && cd build

cmake .. \
  -DMNN_HEXAGON=ON \
  # ... other MNN compilation flags (e.g., cross-compiling for Android)

make -j8
```

*Note: If you have already exported `HEXAGON_SDK_ROOT` in your environment, you can omit the `-DHEXAGON_SDK_ROOT` CMake argument.*

### 3. Deployment and Execution

To run the compiled MNN with the Hexagon backend on an Android device:

1. Push your compiled MNN executable and libraries to the device.
2. Push the generated `libMNN_htpops.so` and `libMNN_htpops_skel.so` libraries to the device.
3. Configure your `LD_LIBRARY_PATH` so the system can find `libMNN_htpops.so`.
4. Configure the `ADSP_LIBRARY_PATH` environment variable to include the directory containing `libMNN_htpops_skel.so` so the DSP can load the skeleton library.

```bash
# Example on device:
BUNDLE=/data/local/tmp/hexagon_libs
export LD_LIBRARY_PATH="/system/lib64:/vendor/lib64:/system/vendor/lib64:$BUNDLE"
export ADSP_LIBRARY_PATH="$BUNDLE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"
```

The `LD_LIBRARY_PATH` order is significant on Android. MNN's FastRPC stub must use the device implementation at
`/vendor/lib64/libcdsprpc.so`, while common Android dependencies such as Binder must resolve from `/system/lib64`
first. Do not put the bundle first when it contains an SDK or QAIRT copy of `libcdsprpc.so`; those SDK libraries may
only contain stubbed control routines and fail to obtain the effective DSP domain. Putting `/vendor/lib64` before
`/system/lib64` can instead mix vendor and system Binder ABIs and cause linker errors.

This backend uses MNN's own FastRPC pair, `libMNN_htpops.so` on the Android host and
`libMNN_htpops_skel.so` (or the architecture-specific `libMNN_htpops_skelV*.so`) on cDSP. It does not use QNN's
`libQnnHtpV*Stub.so` or `libQnnHtpV*Skel.so`.

### Validate a Vision model

Copy `htp-ops-lib/tests/run_android_visual.sh` into the deployment directory and run it from Termux:

```bash
cd /path/to/hexagon_bundle
./run_android_visual.sh /path/to/model /path/to/reference visual-hexagon.log
```

The reference directory must contain `input.mnn` and `output.mnn`. The script selects backend type 10 for both the
primary and backup backend, so an unsupported operator cannot silently fall back to CPU. It also requires evidence
that the architecture-specific skel was opened, the DSP capability query succeeded, all 1084 commands in the tested
Qwen3-VL Vision graph were submitted, and `VISION_FLASH_ATTENTION_FP16` was profiled.

`ModuleBasic.out` uses a fixed max-error check of one percent. That check is useful for short FP32 graphs but can be
too strict for a deep FP16 graph because it ignores error distribution. Preserve its output, then additionally compare
the generated text tensors against the float32 CPU references:

```bash
python3 source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/compare_f32.py \
  visual-output.0.f32 output/0_0.txt --actual-text --min-cosine 0.999 --max-nrmse 0.04
python3 source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/compare_f32.py \
  visual-output.1.f32 output/0_1.txt --actual-text --min-cosine 0.999 --max-nrmse 0.04
```

For the Qwen3-VL-2B FP16 four-token reference used during v79 bring-up, the measured results were:

| Output | Max error | RMS error | NRMSE | Cosine |
| --- | ---: | ---: | ---: | ---: |
| `image_embeds` | 0.0773752 | 0.00497166 | 0.0234260 | 0.99983118 |
| `deepstack_feature` | 0.272133 | 0.0220347 | 0.0331069 | 0.99949445 |

Both repeated runs were bit-identical and contained no NaN or Inf. Always clear and inspect `logcat` around the run;
a successful process exit alone does not rule out an RPC or cDSP failure.

### Simulator versus v79 device numerics

Use the same extracted graph and non-uniform input for simulator/device comparisons. On the first 52-command Vision
block, v79 device versus simulator measured max error 0.00390625 and RMS error 0.00018939. Both paths had nearly the
same difference from the FP32 CPU reference (max error 0.102395 and RMS error about 0.00650). This rules out a large
v79 ISS-versus-HMX-device discrepancy for that block.

Subgraph bisection showed that Attention, FC1, and FC1+GELU each stayed below 0.0014 RMS, while the complete MLP rose
to 0.00629 RMS. Feeding the device GELU activation into the CPU FC2 reproduced 0.00625 RMS, whereas running HMX FC2
and CPU FC2 on the same activation differed by about 0.00102 RMS. The dominant effect is therefore amplification of
small FP16 intermediate differences by the 4096-to-1024 projection, not a different HMX numerical model on hardware.

To reduce the CPU-reference gap, first improve the upstream FP16 kernels that create those activation differences
(especially Conv1x1 accumulation/conversion, LayerNorm, and GELU approximation), and validate each change with the
same subgraph input. Retaining selected intermediates at higher precision would reduce rounding further but changes
the backend tensor contract and increases memory traffic.

The Vision path now uses the dedicated `VISION_FLASH_ATTENTION_FP16` command. It uses the existing worker-pooled HVX
K/V packer to produce the block-256 layout consumed by the HMX/HVX `sync_attention` kernel, and processes queries in
blocks of 64. Its workspace is allocated by the host-side MNN dynamic allocator and passed as a second DSP output,
avoiding a DSP `malloc`/`free` in every Vision layer. The workspace base and all internal regions must remain 128-byte
aligned; aligned offsets from an unaligned base can produce incorrect HMX results without returning an execution
error. Head dimensions that are not multiples of 64 continue to use the scalar `VISION_ATTENTION_FP16` correctness
path.

Vision execution is kept separate from the LLM KV-cache path. It must not update `seq_current`/`seq_add` or add the
LLM page-table input, even when the runtime exposes a non-null `KVMeta` while the multimodal module is executing.
