#!/usr/bin/env bash

# Run CTest on one K3 core class. A100 execution is re-launched before CTest or
# any test binary can establish RVV state.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
BUILD_DIR="${1:?usage: run-tests.sh BUILD_DIR [OUTPUT_FILE]}"
CORE_TYPE="${CORE_TYPE:-x100}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OMP_PLACES="${OMP_PLACES:-cores}"
OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export CORE_TYPE OMP_NUM_THREADS OMP_PLACES OMP_PROC_BIND

if [[ "${CORE_TYPE}" != "x100" && "${CORE_TYPE}" != "a100" ]]; then
  printf 'error: CORE_TYPE must be x100 or a100\n' >&2
  exit 1
fi

if [[ "${CORE_TYPE}" == "a100" && "${GENDIL_K3_A100_PROCESS:-0}" != "1" ]]; then
  if ! command -v ai >/dev/null 2>&1; then
    printf 'error: CORE_TYPE=a100 requires the k3_ai ai launcher\n' >&2
    exit 1
  fi
  exec ai env GENDIL_K3_A100_PROCESS=1 CORE_TYPE="${CORE_TYPE}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" OMP_PLACES="${OMP_PLACES}" \
    OMP_PROC_BIND="${OMP_PROC_BIND}" bash "$0" "$@"
fi

compiler_path="$(
  awk -F= '/^CMAKE_CXX_COMPILER:FILEPATH=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)"
compiler_name="$(basename -- "${compiler_path:-unknown-compiler}")"
OUTPUT_FILE="${2:-${GENDIL_ROOT}/results/spacemit-k3/${compiler_name}/${CORE_TYPE}/tests.txt}"
mkdir -p "$(dirname -- "${OUTPUT_FILE}")"

{
  printf 'commit=%s\n' "$(git -C "${GENDIL_ROOT}" rev-parse HEAD)"
  printf 'core_type=%s\n' "${CORE_TYPE}"
  printf 'a100_launcher=%s\n' "${GENDIL_K3_A100_PROCESS:-0}"
  printf 'omp_num_threads=%s\n' "${OMP_NUM_THREADS}"
  printf 'omp_places=%s\n' "${OMP_PLACES}"
  printf 'omp_proc_bind=%s\n' "${OMP_PROC_BIND}"
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
} 2>&1 | tee "${OUTPUT_FILE}"
