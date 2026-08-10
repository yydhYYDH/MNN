# QuRT HMX Mock Simulator Test

This test boots the Qualcomm QuRT v79 model in `hexagon-sim`, loads a scalar-HMX-mock
`libMNN_htpops_skel.so`, and runs a `32 x 32 x 32` q4/FP16 MatMul through the skel's VTCM, DMA,
worker-pool, and NC64 store paths. It checks every logical output and reports both processor cycles and the 19.2 MHz
QuRT timer.

QuRT, `runelf.pbn`, `qurt_model.so`, `run_main_on_hexagon_sim`, and `hexagon-sim` come from the Qualcomm Hexagon SDK.
This directory only contains the MNN test runner and build orchestration; it does not modify or rebuild the SDK.

Run from any directory:

```bash
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
source/backend/hexagon/htp-ops-lib/tests/qurt-simulator/run.sh
```

Optional environment variables:

- `HEXAGON_TOOLS_ROOT`: select a specific installed Hexagon toolchain; otherwise the script selects one under the SDK.
- `MNN_QURT_SIM_BUILD_DIR`: override the ignored build directory (default: `.artifacts/hexagon-qurt-simulator`).
- `MNN_QURT_SIM_JOBS`: parallel build jobs (default: `4`).

The reported simulator time measures the scalar mock running inside the instruction-set simulator. It is useful for
regression testing, but it is not representative of real HMX hardware performance.
