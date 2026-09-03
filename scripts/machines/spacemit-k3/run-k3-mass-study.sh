#!/usr/bin/env bash

# Run the empirical placement probe before comparing the K3 heterogeneous mass
# operator with the existing host mass operator. This process must start on
# X100 and must not be wrapped with ai.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
BUILD_DIR="${1:?usage: run-k3-mass-study.sh BUILD_DIR [OUTPUT_FILE]}"
PROBE="${BUILD_DIR}/tools/spacemit-k3/k3-heterogeneous-openmp-probe"
MASS="${BUILD_DIR}/tools/spacemit-k3/k3-heterogeneous-openmp-mass"
SHARE="${GENDIL_K3_A100_SHARE:-50}"
compiler_path="$(
  awk -F= '/^CMAKE_CXX_COMPILER:[^=]*=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)"
compiler_name="$(basename -- "${compiler_path:-unknown-compiler}")"
OUTPUT_FILE="${2:-${GENDIL_ROOT}/results/spacemit-k3/${compiler_name}/mass-study-$(git -C "${GENDIL_ROOT}" rev-parse --short HEAD)-share${SHARE}.txt}"

if [[ ! "${SHARE}" =~ ^(0|[1-9][0-9]?)$|^100$ ]]; then
  printf 'error: GENDIL_K3_A100_SHARE must be an integer in [0,100]\n' >&2
  exit 1
fi
if [[ ! -x "${PROBE}" || ! -x "${MASS}" ]]; then
  printf 'error: K3 probe/mass binaries are missing; rebuild with K3_ENABLE_EXPERIMENTS=ON\n' >&2
  exit 1
fi
if [[ ! -w /proc/set_ai_thread ]]; then
  printf 'error: /proc/set_ai_thread is not writable\n' >&2
  exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_FILE}")"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'commit=%s\n' "$(git -C "${GENDIL_ROOT}" rev-parse HEAD)"
  printf 'build_dir=%s\n' "${BUILD_DIR}"
  printf 'a100_share=%s\n' "${SHARE}"
  printf '%s\n' 'placement_probe_begin'
  env -u OMP_THREAD_LIMIT -u GOMP_CPU_AFFINITY -u KMP_AFFINITY \
    OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
    "${PROBE}"
  printf '%s\n' 'placement_probe_end'
  printf '%s\n' 'mass_comparison_begin'
  env -u OMP_THREAD_LIMIT -u GOMP_CPU_AFFINITY -u KMP_AFFINITY \
    GENDIL_K3_A100_SHARE="${SHARE}" \
    OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
    "${MASS}"
  printf '%s\n' 'mass_comparison_end'
} 2>&1 | tee "${OUTPUT_FILE}"

status=${PIPESTATUS[0]}
if [[ ${status} -ne 0 ]]; then
  printf 'K3 mass study failed; see %s\n' "${OUTPUT_FILE}" >&2
  exit "${status}"
fi
printf 'K3 mass study passed; result: %s\n' "${OUTPUT_FILE}"
