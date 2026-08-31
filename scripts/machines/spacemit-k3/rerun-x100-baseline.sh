#!/usr/bin/env bash

# Rebuild and rerun the complete GCC/Clang X100 baseline after pulling K3
# fixes. Individual failures do not prevent the remaining evidence from being
# collected; the script returns nonzero after all requested steps finish.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
GCC_CXX="${GCC_CXX:-/usr/bin/g++-15}"
CLANG_CXX="${CLANG_CXX:-/home/lin32/opt/llvm-main/bin/clang++}"
RUN_GCC="${RUN_GCC:-1}"
RUN_CLANG="${RUN_CLANG:-1}"
RESULT_TAG="${RESULT_TAG:-rerun-$(git -C "${GENDIL_ROOT}" rev-parse --short HEAD)}"

if [[ ! "${RESULT_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'error: RESULT_TAG may contain only letters, digits, dot, underscore, and hyphen\n' >&2
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

run_step()
{
  local description="$1"
  shift
  printf '\n===== %s =====\n' "${description}"
  if "$@"; then
    printf 'PASS: %s\n' "${description}"
    return 0
  else
    status=$?
    failures=$((failures + 1))
    printf 'FAIL (%s): %s\n' "${status}" "${description}" >&2
    return "${status}"
  fi
}

run_compiler()
{
  local compiler_label="$1"
  local compiler_path="$2"
  local build_dir="$3"
  local result_root="${GENDIL_ROOT}/results/spacemit-k3/${compiler_label}"

  if ! run_step "Build ${compiler_label}" \
      env CXX="${compiler_path}" BUILD_DIR="${build_dir}" \
      "${SCRIPT_DIR}/build.sh"; then
    printf 'Skipping %s execution because its build failed.\n' \
      "${compiler_label}" >&2
    return
  fi

  run_step "CTest ${compiler_label} X100" \
    env CORE_TYPE=x100 \
    "${SCRIPT_DIR}/run-tests.sh" \
    "${build_dir}" "${result_root}/x100/tests-${RESULT_TAG}.txt" || true

  run_step "Range smoke ${compiler_label} X100" \
    env BENCHMARK_MODE=smoke CORE_TYPE=x100 \
    "${SCRIPT_DIR}/run-range-benchmarks.sh" \
    "${build_dir}" "${result_root}/x100-smoke-${RESULT_TAG}" || true
}

if [[ "${RUN_GCC}" == "1" ]]; then
  run_compiler "gcc15" "${GCC_CXX}" "${GENDIL_ROOT}/build-k3-g++-15"
fi

if [[ "${RUN_CLANG}" == "1" ]]; then
  run_compiler "clang" "${CLANG_CXX}" "${GENDIL_ROOT}/build-k3-clang++"
fi

if [[ ${failures} -ne 0 ]]; then
  printf '\n%s baseline step(s) failed; commit all generated logs for review.\n' \
    "${failures}" >&2
  exit 1
fi

printf '\nAll requested X100 baseline steps passed.\n'
