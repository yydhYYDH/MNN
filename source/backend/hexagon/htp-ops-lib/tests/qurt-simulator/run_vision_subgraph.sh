#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
MNN_BUILD_DIR="${MNN_HEXAGON_BUILD_DIR:-${REPO_ROOT}/.artifacts/hexagon-x86-offline}"
VISION_MODEL_DIR="${MNN_VISION_MODEL_DIR:-}"
SUBGRAPH="${1:-block}"
BUILD_ROOT="${MNN_QURT_SIM_BUILD_DIR:-${REPO_ROOT}/.artifacts/hexagon-vision-${SUBGRAPH}}"
HOST_BUILD_DIR="${BUILD_ROOT}/host"
MODEL_FILE="${BUILD_ROOT}/visual.mnn"
REFERENCE_FILE="${BUILD_ROOT}/cpu-reference.f32"
REQUEST_FILE="${BUILD_ROOT}/request.bin"

if [[ "${SUBGRAPH}" != "matmul" && "${SUBGRAPH}" != "block" ]]; then
    echo "Usage: $0 [matmul|block]" >&2
    exit 2
fi
if [[ -z "${VISION_MODEL_DIR}" || ! -f "${VISION_MODEL_DIR}/visual.mnn" || \
      ! -f "${VISION_MODEL_DIR}/visual.mnn.weight" ]]; then
    echo "MNN_VISION_MODEL_DIR must contain visual.mnn and visual.mnn.weight" >&2
    exit 2
fi
if [[ ! -f "${MNN_BUILD_DIR}/libMNN.so" || ! -f "${MNN_BUILD_DIR}/express/libMNN_Express.so" ]]; then
    echo "MNN_HEXAGON_BUILD_DIR must contain the x86 MNN_HEXAGON_OFFLINE_RPC build" >&2
    exit 2
fi

cmake -S "${SCRIPT_DIR}" -B "${HOST_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_OFFLINE_RPC_HOST_ONLY=ON \
    "-DMNN_OFFLINE_RPC_MNN_BUILD_DIR=${MNN_BUILD_DIR}"
cmake --build "${HOST_BUILD_DIR}" \
    --target mnn_htp_mnn_matmul_record mnn_htp_mnn_variable_dump mnn_htp_offline_rpc_host \
    -j"${MNN_QURT_SIM_JOBS:-4}"

mkdir -p "${BUILD_ROOT}"
ln -sfn "${VISION_MODEL_DIR}/visual.mnn.weight" "${MODEL_FILE}.weight"

if [[ "${SUBGRAPH}" == "matmul" ]]; then
    "${HOST_BUILD_DIR}/mnn_htp_mnn_variable_dump" extract-conv \
        "${VISION_MODEL_DIR}/visual.mnn" \
        /mlp/linear_fc2/Add_output_0__matmul_converted "${MODEL_FILE}" 4
    "${HOST_BUILD_DIR}/mnn_htp_mnn_matmul_record" cpu-model "${MODEL_FILE}" "${REFERENCE_FILE}"
    MNN_HEXAGON_OFFLINE_RPC_PATH="${REQUEST_FILE}" \
        "${HOST_BUILD_DIR}/mnn_htp_mnn_matmul_record" record-model "${MODEL_FILE}"
    ABS_TOLERANCE=0.01
    RMS_TOLERANCE=0.01
else
    "${HOST_BUILD_DIR}/mnn_htp_mnn_variable_dump" extract-block \
        "${VISION_MODEL_DIR}/visual.mnn" /Reshape_1_output_0 /Add_8_output_0 "${MODEL_FILE}" 4
    "${HOST_BUILD_DIR}/mnn_htp_mnn_matmul_record" cpu-block "${MODEL_FILE}" "${REFERENCE_FILE}"
    MNN_HEXAGON_OFFLINE_RPC_PATH="${REQUEST_FILE}" \
        "${HOST_BUILD_DIR}/mnn_htp_mnn_matmul_record" record-block "${MODEL_FILE}"
    ABS_TOLERANCE=0.125
    RMS_TOLERANCE=0.01
fi

MNN_OFFLINE_RPC_REQUEST="${REQUEST_FILE}" \
MNN_OFFLINE_RPC_REFERENCE="${REFERENCE_FILE}" \
MNN_OFFLINE_RPC_ABS_TOLERANCE="${ABS_TOLERANCE}" \
MNN_OFFLINE_RPC_RMS_TOLERANCE="${RMS_TOLERANCE}" \
MNN_QURT_SIM_BUILD_DIR="${BUILD_ROOT}" \
    "${SCRIPT_DIR}/run.sh"

echo "Vision ${SUBGRAPH} offline RPC test passed"
echo "Model: ${MODEL_FILE}"
echo "Request: ${REQUEST_FILE}"
echo "CPU reference: ${REFERENCE_FILE}"
