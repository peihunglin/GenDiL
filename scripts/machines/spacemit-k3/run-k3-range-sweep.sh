#!/usr/bin/env bash

# Collect all 18 range benchmarks with the opt-in heterogeneous K3 policy for
# both compilers. The binaries must be built with K3 experiments enabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
GCC_BUILD_DIR="${GCC_BUILD_DIR:-${GENDIL_ROOT}/build-k3-g++-15}"
CLANG_BUILD_DIR="${CLANG_BUILD_DIR:-${GENDIL_ROOT}/build-k3-clang++}"
SHARES="${SHARES:-25 50 75}"
RUN_GCC="${RUN_GCC:-1}"
RUN_CLANG="${RUN_CLANG:-1}"
RESULT_TAG="${RESULT_TAG:-$(git -C "${GENDIL_ROOT}" rev-parse --short HEAD)}"

if [[ ! "${RESULT_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'error: RESULT_TAG contains unsupported characters\n' >&2
  exit 1
fi
if [[ "${RUN_GCC}" == "0" && "${RUN_CLANG}" == "0" ]]; then
  printf 'error: at least one compiler must be enabled\n' >&2
  exit 1
fi

failures=0
run_compiler()
{
  local label="$1"
  local build_dir="$2"
  local share
  local output_root="${GENDIL_ROOT}/results/spacemit-k3/${label}/range-mixed-${RESULT_TAG}"

  for share in ${SHARES}; do
    if ! CORE_TYPE=mixed BENCHMARK_MODE=performance \
        GENDIL_K3_A100_SHARE="${share}" \
        "${SCRIPT_DIR}/run-range-benchmarks.sh" \
        "${build_dir}" "${output_root}/share-${share}"; then
      failures=$((failures + 1))
    fi
  done
}

if [[ "${RUN_GCC}" == "1" ]]; then
  run_compiler gcc15 "${GCC_BUILD_DIR}"
fi
if [[ "${RUN_CLANG}" == "1" ]]; then
  run_compiler clang "${CLANG_BUILD_DIR}"
fi

if [[ ${failures} -ne 0 ]]; then
  printf '%s mixed range sweep(s) failed\n' "${failures}" >&2
  exit 1
fi
printf 'K3 mixed range sweep passed for all requested runs\n'
