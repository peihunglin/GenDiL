#!/usr/bin/env bash

# Collect the K3 heterogeneous mass comparison for both compilers and several
# X100/A100 work splits. Each run repeats placement validation before applying
# the operator, so a failed placement cannot be hidden by later results.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
GCC_BUILD_DIR="${GCC_BUILD_DIR:-${GENDIL_ROOT}/build-k3-g++-15}"
CLANG_BUILD_DIR="${CLANG_BUILD_DIR:-${GENDIL_ROOT}/build-k3-clang++}"
SHARES="${SHARES:-0 25 50 75 100}"
RUN_GCC="${RUN_GCC:-1}"
RUN_CLANG="${RUN_CLANG:-1}"
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
  local share
  local output_dir="${GENDIL_ROOT}/results/spacemit-k3/${compiler_label}/mass-share-${RESULT_TAG}"

  for share in ${SHARES}; do
    if [[ ! "${share}" =~ ^(0|[1-9][0-9]?)$|^100$ ]]; then
      printf 'error: invalid A100 share: %s\n' "${share}" >&2
      failures=$((failures + 1))
      continue
    fi
    if ! GENDIL_K3_A100_SHARE="${share}" \
        "${SCRIPT_DIR}/run-k3-mass-study.sh" \
        "${build_dir}" "${output_dir}/share-${share}.txt"; then
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
  printf '%s K3 mass share run(s) failed\n' "${failures}" >&2
  exit 1
fi
printf 'K3 mass share sweep passed for all requested runs\n'
