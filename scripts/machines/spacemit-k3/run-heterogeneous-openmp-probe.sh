#!/usr/bin/env bash

# Study the K3 GEMM-style heterogeneous OpenMP mechanism. Run normally on X100;
# do not invoke this script through ai.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
BUILD_DIR="${1:?usage: run-heterogeneous-openmp-probe.sh BUILD_DIR [OUTPUT_FILE]}"
PROBE="${BUILD_DIR}/tools/spacemit-k3/k3-heterogeneous-openmp-probe"
compiler_path="$(
  awk -F= '/^CMAKE_CXX_COMPILER:[^=]*=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)"
compiler_name="$(basename -- "${compiler_path:-unknown-compiler}")"
OUTPUT_FILE="${2:-${GENDIL_ROOT}/results/spacemit-k3/${compiler_name}/heterogeneous-openmp-probe-$(git -C "${GENDIL_ROOT}" rev-parse --short HEAD).txt}"

if [[ ! -x "${PROBE}" ]]; then
  printf 'error: K3 probe executables are missing; rerun the K3 build script\n' >&2
  exit 1
fi
if [[ ! -w /proc/set_ai_thread ]]; then
  printf 'error: /proc/set_ai_thread is not writable\n' >&2
  exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_FILE}")"

set +e
{
  set -e
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'commit=%s\n' "$(git -C "${GENDIL_ROOT}" rev-parse HEAD)"
  printf 'build_dir=%s\n' "${BUILD_DIR}"
  printf 'compiler=%s\n' "${compiler_path}"
  "${compiler_path}" --version
  uname -a
  if [[ -r /proc/sys/abi/riscv_v_default_allow ]]; then
    printf 'riscv_v_default_allow=%s\n' \
      "$(< /proc/sys/abi/riscv_v_default_allow)"
  fi
  ls -l /proc/set_ai_thread
  printf '%s\n' 'probe_output_begin'
  set +e
  env -u OMP_THREAD_LIMIT -u GOMP_CPU_AFFINITY -u KMP_AFFINITY \
    OMP_NUM_THREADS=16 \
    OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=FALSE \
    "${PROBE}"
  probe_status=$?
  printf '%s\n' 'probe_output_end'
  exit "${probe_status}"
} 2>&1 | tee "${OUTPUT_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
set -e

status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
if [[ ${tee_status} -ne 0 ]]; then
  printf 'failed to write probe output: %s\n' "${OUTPUT_FILE}" >&2
  exit "${tee_status}"
fi

if [[ ${status} -ne 0 ]]; then
  printf 'heterogeneous OpenMP probe failed; see %s\n' "${OUTPUT_FILE}" >&2
  exit "${status}"
fi

printf 'heterogeneous OpenMP probe passed; result: %s\n' "${OUTPUT_FILE}"
