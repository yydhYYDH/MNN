#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
BUILD_ROOT="${MNN_QURT_SIM_BUILD_DIR:-${REPO_ROOT}/.artifacts/hexagon-mnn-matmul-e2e}"
MNN_BUILD_DIR="${MNN_HEXAGON_BUILD_DIR:-${REPO_ROOT}/.artifacts/hexagon-x86-offline}"
HOST_BUILD_DIR="${BUILD_ROOT}/host"
MODEL_FILE="${BUILD_ROOT}/matmul-32.mnn"
REFERENCE_FILE="${BUILD_ROOT}/matmul-32.cpu.bin"
REQUEST_FILE="${BUILD_ROOT}/matmul-32.rpc.bin"
MATMUL_K="${MNN_MATMUL_K:-32}"

if [[ ! -f "${MNN_BUILD_DIR}/libMNN.so" || ! -f "${MNN_BUILD_DIR}/express/libMNN_Express.so" ]]; then
    echo "MNN_HEXAGON_BUILD_DIR must contain the x86 MNN_HEXAGON_OFFLINE_SIMULATOR build" >&2
    exit 2
fi

cmake -S "${SCRIPT_DIR}" -B "${HOST_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_OFFLINE_RPC_HOST_ONLY=ON \
    "-DMNN_OFFLINE_RPC_MNN_BUILD_DIR=${MNN_BUILD_DIR}"
cmake --build "${HOST_BUILD_DIR}" --target mnn_htp_mnn_matmul_record mnn_htp_offline_rpc_host \
    -j"${MNN_QURT_SIM_JOBS:-4}"

mkdir -p "${BUILD_ROOT}"
"${HOST_BUILD_DIR}/mnn_htp_mnn_matmul_record" create "${MODEL_FILE}" "${MATMUL_K}"
"${HOST_BUILD_DIR}/mnn_htp_mnn_matmul_record" cpu "${MODEL_FILE}" "${REFERENCE_FILE}"
MNN_HEXAGON_OFFLINE_RPC_PATH="${REQUEST_FILE}" \
    "${HOST_BUILD_DIR}/mnn_htp_mnn_matmul_record" record "${MODEL_FILE}"

MNN_OFFLINE_RPC_REQUEST="${REQUEST_FILE}" \
MNN_OFFLINE_RPC_REFERENCE="${REFERENCE_FILE}" \
MNN_QURT_SIM_BUILD_DIR="${BUILD_ROOT}" \
    "${SCRIPT_DIR}/run_qurt_simulator.sh"

echo "MNN MatMul offline RPC end-to-end test passed"
echo "Model: ${MODEL_FILE}"
echo "Request: ${REQUEST_FILE}"
echo "CPU reference: ${REFERENCE_FILE}"
