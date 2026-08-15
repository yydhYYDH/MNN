# Hexagon Offline QuRT Simulator

This directory provides a test-only offline validation flow for the Hexagon backend:

```text
MNN graph -> x86 Hexagon backend -> offline RPC request
          -> QuRT v79 simulator -> HMX mock -> response
```

It validates command serialization, buffer relocation, DSP dispatch, and output correctness. It does not represent real HMX performance or model FastRPC, DMA-BUF, or cache-coherency overhead.

## Requirements

- Hexagon SDK, specified with `HEXAGON_SDK_ROOT`;
- v79 Hexagon toolchain;
- native MNN build for `.mnn` graph validation;
- Linux/WSL runtime libraries required by the QuRT simulator.

The runner currently supports `DSP_ARCH=v79` only.

## Quick smoke test

Run the synthetic MatMul smoke test:

```bash
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/run_qurt_simulator.sh
```

The script builds the DSP mock, QuRT runner, and host tool, creates a 32x32x32 MatMul request, executes it in the simulator, and verifies the response.

## Validate an MNN graph

Build native MNN with the offline simulator enabled:

```bash
cmake -S . -B .artifacts/hexagon-x86-offline -G Ninja \
    -DMNN_HEXAGON=ON \
    -DMNN_HEXAGON_OFFLINE_SIMULATOR=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_BUILD_TEST=ON \
    -DCMAKE_CXX_FLAGS=-D__fp16=short
cmake --build .artifacts/hexagon-x86-offline -j4
```

Run the fixed MatMul regression:

```bash
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/verify_hexagon_matmul.sh
```

The fixed script is for regression coverage. For arbitrary `.mnn` graphs, use `mnn_htp_graph_record`:

```bash
cmake -S source/backend/hexagon/htp-ops-lib/tests/qurt-simulator \
    -B .artifacts/hexagon-graph-host -G Ninja \
    -DMNN_OFFLINE_RPC_HOST_ONLY=ON \
    -DMNN_OFFLINE_RPC_MNN_BUILD_DIR=.artifacts/hexagon-x86-offline
cmake --build .artifacts/hexagon-graph-host \
    --target mnn_htp_graph_record mnn_htp_op_graph_gen -j4
```

Prepare an input manifest for an existing `.mnn` graph. Each line contains one model input name and its binary file:

```text
input_tensor_name /path/to/input.bin
```

Generate a CPU reference, record the graph, and run the simulator:

```bash
GRAPH=.artifacts/my-graph
mkdir -p "$GRAPH"

.artifacts/hexagon-graph-host/mnn_htp_graph_record \
    cpu model.mnn inputs.txt output "$GRAPH/reference.f32"

MNN_HEXAGON_OFFLINE_RPC_PATH="$GRAPH/request.bin" \
.artifacts/hexagon-graph-host/mnn_htp_graph_record \
    record model.mnn inputs.txt output

MNN_OFFLINE_RPC_REQUEST="$GRAPH/request.bin" \
MNN_OFFLINE_RPC_REFERENCE="$GRAPH/reference.f32" \
MNN_QURT_SIM_BUILD_DIR="$GRAPH/sim" \
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/run_qurt_simulator.sh
```

`record` compares one selected output. Multi-output graphs require separate output handling. Input file sizes and model tensor shapes are checked before recording.

## Generate single-op graphs

`mnn_htp_op_graph_gen` currently supports:

```text
flash-attention
softmax
add
```

For example:

```bash
.artifacts/hexagon-graph-host/mnn_htp_op_graph_gen \
    softmax model.mnn inputs.txt
```

Validate the generated model and manifest with the general graph workflow above. When adding support for another operator, prefer using an existing exported `.mnn` graph and manifest; extend the generator only when no suitable graph is available.

## Common configuration

- `HEXAGON_TOOLS_ROOT`: select the Hexagon toolchain;
- `MNN_HEXAGON_BUILD_DIR`: native MNN build directory;
- `MNN_QURT_SIM_BUILD_DIR`: simulator build directory;
- `MNN_OFFLINE_RPC_REQUEST`: use an existing request instead of creating a synthetic one;
- `MNN_OFFLINE_RPC_REFERENCE`: CPU reference file for output comparison;
- `MNN_MATMUL_K`: K dimension for the MatMul regression, default `32`;
- `MNN_OFFLINE_RPC_ABS_TOLERANCE`: absolute error tolerance, default `0.01`.

## Current limits

- QuRT v79 simulator only;
- up to 64 buffers and 65536 commands per request, with a 64 KiB limit per command;
- the runtime flushes every 4096 commands, while the offline request accumulates and replays all groups in order;
- the simulator uses an HMX mock and does not provide real hardware performance data;
- supported operators depend on the Hexagon DSP mock and the current MNN Hexagon backend implementation.
