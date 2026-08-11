#!/system/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 MODEL_DIR REFERENCE_DIR [LOG_FILE]" >&2
    exit 2
fi

MODEL_DIR="$1"
REFERENCE_DIR="$2"
LOG_FILE="${3:-visual-hexagon.log}"
BUNDLE="${MNN_HEXAGON_BUNDLE:-$(pwd)}"

for file in ModuleBasic.out libMNN.so libMNN_htpops.so; do
    if [ ! -f "${BUNDLE}/${file}" ]; then
        echo "Missing ${BUNDLE}/${file}" >&2
        exit 3
    fi
done
if [ ! -f "${BUNDLE}/libMNN_htpops_skel.so" ] && ! ls "${BUNDLE}"/libMNN_htpops_skelV*.so >/dev/null 2>&1; then
    echo "Missing MNN cDSP skeleton in ${BUNDLE}" >&2
    exit 3
fi

export LD_LIBRARY_PATH="/system/lib64:/vendor/lib64:/system/vendor/lib64:${BUNDLE}"
export ADSP_LIBRARY_PATH="${BUNDLE};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"

set +e
"${BUNDLE}/ModuleBasic.out" "${MODEL_DIR}/visual.mnn" "${REFERENCE_DIR}" 128 10 1 4 0 >"${LOG_FILE}" 2>&1
run_status=$?
set -e
cat "${LOG_FILE}"
if [ "${run_status}" -ne 0 ]; then
    exit "${run_status}"
fi

grep -q "skel arch verified" "${LOG_FILE}"
grep -q "vectorSize=.*vtcmSize=.*maxThreads=.*hvxArch=" "${LOG_FILE}"
grep -q "max commands/group: 1084" "${LOG_FILE}"
grep -q "DSPOpType VISION_FLASH_ATTENTION_FP16" "${LOG_FILE}"
if grep -Eq "unsupported op|resize failed|execute_command_group_profile failed|Error in forward|Fatal signal" "${LOG_FILE}"; then
    echo "Hexagon execution failure found in ${LOG_FILE}" >&2
    exit 4
fi

echo "visual.mnn completed through the Hexagon backend; run compare_f32.py on output/0_0.txt and output/0_1.txt"
