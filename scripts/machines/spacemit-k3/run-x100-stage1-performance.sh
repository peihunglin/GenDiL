#!/usr/bin/env bash

# Collect stable 1/2/4/8-thread X100 results for both configured compilers.
# Builds and correctness tests must already pass through the baseline rerun.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
GCC_BUILD_DIR="${GCC_BUILD_DIR:-${GENDIL_ROOT}/build-k3-g++-15}"
CLANG_BUILD_DIR="${CLANG_BUILD_DIR:-${GENDIL_ROOT}/build-k3-clang++}"
RUN_GCC="${RUN_GCC:-1}"
RUN_CLANG="${RUN_CLANG:-1}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8}"
RESULT_TAG="${RESULT_TAG:-$(git -C "${GENDIL_ROOT}" rev-parse --short HEAD)}"

if [[ ! "${RESULT_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'error: RESULT_TAG contains unsupported characters\n' >&2
  exit 1
fi
if [[ ! "${RUN_GCC}" =~ ^[01]$ || ! "${RUN_CLANG}" =~ ^[01]$ ]]; then
  printf 'error: RUN_GCC and RUN_CLANG must be 0 or 1\n' >&2
  exit 1
fi
if [[ "${RUN_GCC}" == "0" && "${RUN_CLANG}" == "0" ]]; then
  printf 'error: at least one compiler must be enabled\n' >&2
  exit 1
fi

failures=0

run_compiler()
{
  local compiler_label="$1"
  local build_dir="$2"
  local threads
  local output_dir

  if [[ ! -x "${build_dir}/benchmarks/range-benchmark-sol-3d" ]]; then
    printf 'error: range benchmarks are not built in %s\n' "${build_dir}" >&2
    failures=$((failures + 1))
    return
  fi

  for threads in ${THREAD_COUNTS}; do
    if [[ ! "${threads}" =~ ^[1-9][0-9]*$ ]]; then
      printf 'error: invalid thread count: %s\n' "${threads}" >&2
      failures=$((failures + 1))
      continue
    fi

    output_dir="${GENDIL_ROOT}/results/spacemit-k3/${compiler_label}/x100-performance-${threads}t-${RESULT_TAG}"
    printf '\n===== %s: %s X100 threads =====\n' \
      "${compiler_label}" "${threads}"
    if ! BENCHMARK_MODE=performance CORE_TYPE=x100 \
        OMP_NUM_THREADS="${threads}" \
        "${SCRIPT_DIR}/run-range-benchmarks.sh" \
        "${build_dir}" "${output_dir}"; then
      failures=$((failures + 1))
    fi
  done
}

if [[ "${RUN_GCC}" == "1" ]]; then
  run_compiler "gcc15" "${GCC_BUILD_DIR}"
fi
if [[ "${RUN_CLANG}" == "1" ]]; then
  run_compiler "clang" "${CLANG_BUILD_DIR}"
fi

if [[ ${failures} -ne 0 ]]; then
  printf '%s performance sweep(s) failed; commit all logs for review.\n' \
    "${failures}" >&2
  exit 1
fi

printf '\nAll requested Stage 1 performance sweeps passed.\n'
