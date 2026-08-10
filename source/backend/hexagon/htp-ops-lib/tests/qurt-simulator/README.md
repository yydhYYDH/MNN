# QuRT HMX mock simulator tests

These tests boot the Qualcomm QuRT v79 model in `hexagon-sim`, load an HVX-based HMX mock
`libMNN_htpops_skel.so`, and execute the normal DSP command dispatcher. QuRT, `runelf.pbn`, `qurt_model.so`,
`run_main_on_hexagon_sim`, and `hexagon-sim` come from the Qualcomm Hexagon SDK. The tests do not modify the SDK.

The transport is a versioned file-backed offline RPC package. It contains real `DSPCOMMAND::Command` FlatBuffers,
synthetic buffer IDs, sparse 4 KiB buffer chunks, and the final output tensor region. The QuRT runner reconstructs the
full logical buffers and ID-to-pointer mapping, executes each command through the skel, and returns the output and DSP
timing. This validates
serialization, relocation, command dispatch, and DSP execution; it does not model FastRPC, DMA-BUF, or cache-coherency
overhead.

This is a test-only path. `MNN_HEXAGON_OFFLINE_RPC` is disabled by default, and the Vision FP16 command used by the
block test is compiled only with the simulator mock. It does not change the Android FastRPC backend's supported model
contract: production Hexagon LLM inference still requires the documented 4-bit symmetric Transformer C4 export.

## Synthetic command smoke test

The basic test creates one `DSP_OP_MATMUL_Q4A16_FP16` command and verifies every logical output of a 32x32x32 MatMul:

```bash
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/run.sh
```

## MNN graph end-to-end test

First build native MNN with the record-only transport:

```bash
cmake -S . -B .artifacts/hexagon-x86-offline -G Ninja \
    -DMNN_HEXAGON=ON \
    -DMNN_HEXAGON_OFFLINE_RPC=ON \
    -DMNN_BUILD_TEST=ON
cmake --build .artifacts/hexagon-x86-offline -j4
```

Then run the full test:

```bash
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/run_mnn_matmul.sh
```

The script generates a MatMul graph with deterministic nonuniform inputs, weights, and bias. It validates the CPU
result against an independent scalar formula, records commands through the x86 Hexagon backend, runs them under QuRT
v79, and compares every returned FP16 value with the CPU reference. Set `MNN_MATMUL_K=4096` to cover a Vision-style
projection depth. Model artifacts and logs are stored under the ignored `.artifacts` directory.

## Vision subgraph tests

An exported Vision model with external FP16 weights can be tested without running the complete graph:

```bash
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
export MNN_VISION_MODEL_DIR=/path/to/Qwen3-VL-2B-Instruct-MNN-FP16

# Real M=4, K=4096, N=1024 projection with the model's weights.
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/run_vision_subgraph.sh matmul

# One complete four-token ViT block, including Attention, MLP, and residual paths.
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/run_vision_subgraph.sh block
```

The MatMul test retains the default `0.01` absolute-error gate. The complete FP16 block uses explicit
`max_abs=0.125` and `RMS=0.01` gates because small FP16 errors before the final K=4096 projection can be amplified.
The verifier still reports counts above fixed `0.02` and `0.05` thresholds, the maximum-error element, and all
non-finite values.

Optional environment variables:

- `HEXAGON_TOOLS_ROOT`: select a toolchain under the SDK.
- `MNN_HEXAGON_BUILD_DIR`: select the native record-only MNN build.
- `MNN_QURT_SIM_BUILD_DIR`: override the ignored simulator build directory.
- `MNN_QURT_SIM_JOBS`: parallel build jobs; default `4`.
- `MNN_OFFLINE_RPC_ABS_TOLERANCE`: CPU comparison absolute-error limit; default `0.01`.
- `MNN_OFFLINE_RPC_RMS_TOLERANCE`: optional CPU comparison RMS-error limit.

Simulator timing measures the HVX mock inside the instruction-set simulator and is not representative of real HMX
hardware performance.
