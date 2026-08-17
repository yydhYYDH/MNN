#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${MNN_OFFLINE_RPC_PAYLOAD_TEST_BUILD_DIR:-${SCRIPT_DIR}/.payload-test-build}"
REQUEST_FILE="${BUILD_ROOT}/valid.request.bin"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_ROOT}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DMNN_OFFLINE_RPC_HOST_ONLY=ON
cmake --build "${BUILD_ROOT}" --target mnn_htp_offline_rpc_host -j"${MNN_QURT_SIM_JOBS:-4}"

mkdir -p "${BUILD_ROOT}"
"${BUILD_ROOT}/mnn_htp_offline_rpc_host" create "${REQUEST_FILE}"

expect_rejected() {
    local request_file="$1"
    if "${BUILD_ROOT}/mnn_htp_offline_rpc_host" inspect "${request_file}" >/dev/null 2>&1; then
        echo "Corrupt request was accepted: ${request_file}" >&2
        exit 1
    fi
}

# The built-in request has four buffer descriptors, one command descriptor,
# and therefore starts its command payload at 40 + 4 * 20 + 8 bytes.
COMMAND_OFFSET=128
ZERO_REQUEST="${BUILD_ROOT}/zero-command.request.bin"
cp "${REQUEST_FILE}" "${ZERO_REQUEST}"
dd if=/dev/zero of="${ZERO_REQUEST}" bs=1 seek="${COMMAND_OFFSET}" count=32 conv=notrunc status=none
expect_rejected "${ZERO_REQUEST}"

RANDOM_REQUEST="${BUILD_ROOT}/random-command.request.bin"
cp "${REQUEST_FILE}" "${RANDOM_REQUEST}"
dd if=/dev/urandom of="${RANDOM_REQUEST}" bs=1 seek="${COMMAND_OFFSET}" count=32 conv=notrunc status=none
expect_rejected "${RANDOM_REQUEST}"

echo "Offline RPC command payload validation passed"
