#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTP_OPS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${HTP_OPS_ROOT}/../../../.." && pwd)"
SDK_ROOT="${HEXAGON_SDK_ROOT:-}"
DSP_ARCH="${DSP_ARCH:-v79}"
BUILD_ROOT="${MNN_QURT_SIM_BUILD_DIR:-${REPO_ROOT}/.artifacts/hexagon-qurt-simulator}"

if [[ -z "${SDK_ROOT}" || ! -d "${SDK_ROOT}" ]]; then
    echo "HEXAGON_SDK_ROOT must point to an installed Hexagon SDK" >&2
    exit 2
fi
if [[ "${DSP_ARCH}" != "v79" ]]; then
    echo "This mock HMX runner currently validates v79 only (DSP_ARCH=${DSP_ARCH})" >&2
    exit 2
fi

TOOLS_ROOT="${HEXAGON_TOOLS_ROOT:-}"
if [[ -z "${TOOLS_ROOT}" ]]; then
    for candidate in "${SDK_ROOT}"/tools/HEXAGON_Tools/*; do
        if [[ -x "${candidate}/Tools/bin/hexagon-clang" ]]; then
            TOOLS_ROOT="${candidate}"
        fi
    done
fi
if [[ -z "${TOOLS_ROOT}" || ! -x "${TOOLS_ROOT}/Tools/bin/hexagon-clang" ]]; then
    echo "HEXAGON_TOOLS_ROOT is invalid and no toolchain was found under the SDK" >&2
    exit 2
fi

TOOLS_VERSION="${TOOLS_ROOT##*/}"
TOOLS_MAJOR="${TOOLS_VERSION%%.*}"
PREBUILT_LIB_DIR="hexagon_toolv${TOOLS_MAJOR}_${DSP_ARCH}"
TOOLCHAIN_FILE="${SDK_ROOT}/build/cmake/hexagon_toolchain.cmake"
SKEL_BUILD_DIR="${BUILD_ROOT}/skel"
RUNNER_BUILD_DIR="${BUILD_ROOT}/runner"
HOST_BUILD_DIR="${BUILD_ROOT}/host"
REQUEST_FILE="${RUNNER_BUILD_DIR}/offline_rpc_request.bin"
RESPONSE_FILE="${RUNNER_BUILD_DIR}/offline_rpc_response.bin"

COMMON_CMAKE_ARGS=(
    -G Ninja
    "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
    "-DHEXAGON_SDK_ROOT=${SDK_ROOT}"
    "-DHEXAGON_TOOLS_ROOT=${TOOLS_ROOT}"
    "-DHEXAGON_CMAKE_ROOT=${SDK_ROOT}/build/cmake"
    "-DPREBUILT_LIB_DIR=${PREBUILT_LIB_DIR}"
    "-DDSP_VERSION=${DSP_ARCH}"
    -DQURT_OS=1
    -DSIM_TYPE=sim
    -DCMAKE_BUILD_TYPE=Release
)

cmake -S "${HTP_OPS_ROOT}" -B "${SKEL_BUILD_DIR}" "${COMMON_CMAKE_ARGS[@]}" \
    -DHTP_OPS_SIMULATOR_MOCK_HMX=ON
cmake --build "${SKEL_BUILD_DIR}" --target htp_ops_skel -j"${MNN_QURT_SIM_JOBS:-4}"

SKEL_SO="${SKEL_BUILD_DIR}/ship/libMNN_htpops_skel.so"
if [[ ! -f "${SKEL_SO}" ]]; then
    SKEL_SO="${SKEL_BUILD_DIR}/libMNN_htpops_skel.so"
fi
if [[ ! -f "${SKEL_SO}" ]]; then
    echo "Mock skel was not produced by the DSP build" >&2
    exit 3
fi

cmake -S "${SCRIPT_DIR}" -B "${HOST_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DMNN_OFFLINE_RPC_HOST_ONLY=ON
cmake --build "${HOST_BUILD_DIR}" --target mnn_htp_offline_rpc_host -j"${MNN_QURT_SIM_JOBS:-4}"

if [[ "${MNN_OFFLINE_RPC_ASYMMETRIC_Q4:-0}" == "1" ]]; then
    ASYMMETRIC_REQUEST="${BUILD_ROOT}/q4_asymmetric_request.bin"
    ASYMMETRIC_REFERENCE="${BUILD_ROOT}/q4_asymmetric_reference.bin"
    ASYMMETRIC_CREATE_MODE="create-asymmetric"
    if [[ "${MNN_OFFLINE_RPC_Q4_M:-64}" == "1" ]]; then
        ASYMMETRIC_CREATE_MODE="create-asymmetric-m1"
    fi
    "${HOST_BUILD_DIR}/mnn_htp_offline_rpc_host" "${ASYMMETRIC_CREATE_MODE}" \
        "${ASYMMETRIC_REQUEST}" "${ASYMMETRIC_REFERENCE}"
    MNN_OFFLINE_RPC_REQUEST="${ASYMMETRIC_REQUEST}"
    MNN_OFFLINE_RPC_REFERENCE="${ASYMMETRIC_REFERENCE}"
fi
if [[ "${MNN_OFFLINE_RPC_W8_BLOCK:-0}" == "1" ]]; then
    W8_BLOCK_REQUEST="${BUILD_ROOT}/w8_block_request.bin"
    W8_BLOCK_REFERENCE="${BUILD_ROOT}/w8_block_reference.bin"
    W8_BLOCK_CREATE_MODE="create-w8-block"
    if [[ "${MNN_OFFLINE_RPC_W8_BLOCK_M:-64}" == "1" ]]; then
        W8_BLOCK_CREATE_MODE="create-w8-block-m1"
    fi
    if [[ "${MNN_OFFLINE_RPC_W8_ASYMMETRIC:-0}" == "1" ]]; then
        W8_BLOCK_CREATE_MODE="create-w8-block-asymmetric"
        if [[ "${MNN_OFFLINE_RPC_W8_BLOCK_M:-64}" == "1" ]]; then
            W8_BLOCK_CREATE_MODE="create-w8-block-asymmetric-m1"
        fi
    fi
    "${HOST_BUILD_DIR}/mnn_htp_offline_rpc_host" "${W8_BLOCK_CREATE_MODE}" \
        "${W8_BLOCK_REQUEST}" "${W8_BLOCK_REFERENCE}"
    MNN_OFFLINE_RPC_REQUEST="${W8_BLOCK_REQUEST}"
    MNN_OFFLINE_RPC_REFERENCE="${W8_BLOCK_REFERENCE}"
fi

cmake -S "${SCRIPT_DIR}" -B "${RUNNER_BUILD_DIR}" "${COMMON_CMAKE_ARGS[@]}" \
    "-DMNN_HTPOPS_SKEL=${SKEL_SO}"
cmake --build "${RUNNER_BUILD_DIR}" --target mnn_htp_qurt_runner -j"${MNN_QURT_SIM_JOBS:-4}"
rm -f "${REQUEST_FILE}" "${RESPONSE_FILE}"
if [[ -n "${MNN_OFFLINE_RPC_REQUEST:-}" ]]; then
    cp "${MNN_OFFLINE_RPC_REQUEST}" "${REQUEST_FILE}"
    INSPECT_LOG="${BUILD_ROOT}/offline_rpc_inspect.txt"
    "${HOST_BUILD_DIR}/mnn_htp_offline_rpc_host" inspect "${REQUEST_FILE}" > "${INSPECT_LOG}"
    head -n 1 "${INSPECT_LOG}"
    echo "Full request inspection: ${INSPECT_LOG}"
else
    "${HOST_BUILD_DIR}/mnn_htp_offline_rpc_host" create "${REQUEST_FILE}"
fi

SIMULATOR="${TOOLS_ROOT}/Tools/bin/hexagon-sim"
SIM_COMPAT_DIR="${BUILD_ROOT}/sim-compat"
mkdir -p "${SIM_COMPAT_DIR}"
if ldd "${SIMULATOR}" 2>/dev/null | grep -q 'libncurses.so.5 => not found'; then
    NCURSES_COMPAT="$(ldconfig -p 2>/dev/null | awk '/libncurses.so.6 .*x86-64/{print $NF; exit}')"
    if [[ -z "${NCURSES_COMPAT}" ]]; then
        echo "hexagon-sim requires libncurses.so.5; install it or set LD_LIBRARY_PATH" >&2
        exit 4
    fi
    ln -sfn "${NCURSES_COMPAT}" "${SIM_COMPAT_DIR}/libncurses.so.5"
fi

export LD_LIBRARY_PATH="${SIM_COMPAT_DIR}:${TOOLS_ROOT}/Tools/lib/iss:${LD_LIBRARY_PATH:-}"
SIM_START_SECONDS="${SECONDS}"
cmake --build "${RUNNER_BUILD_DIR}" --target runOnSimulator
SIM_WALL_SECONDS="$((SECONDS - SIM_START_SECONDS))"
if [[ -z "${MNN_OFFLINE_RPC_REQUEST:-}" ]]; then
    "${HOST_BUILD_DIR}/mnn_htp_offline_rpc_host" verify "${RESPONSE_FILE}"
fi
if [[ -n "${MNN_OFFLINE_RPC_REFERENCE:-}" ]]; then
    "${HOST_BUILD_DIR}/mnn_htp_offline_rpc_host" verify-reference "${RESPONSE_FILE}" "${MNN_OFFLINE_RPC_REFERENCE}"
fi
echo "Offline RPC simulator wall time: ${SIM_WALL_SECONDS} s"

LOG_FILE="${BUILD_ROOT}/sim_run_logs.txt"
if [[ ! -f "${LOG_FILE}" ]]; then
    LOG_FILE="${RUNNER_BUILD_DIR}/../sim_run_logs.txt"
fi
if ! grep -q 'offline_rpc_graph: err=0' "${LOG_FILE}" || ! grep -q 'Main() returned 0' "${LOG_FILE}"; then
    echo "QuRT simulator smoke test did not report a clean result: ${LOG_FILE}" >&2
    exit 5
fi

echo "QuRT simulator smoke test passed"
echo "Log: ${LOG_FILE}"
