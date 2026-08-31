#!/usr/bin/env bash

# Configure and build the native K3 correctness and range-benchmark targets.
# Compiler and ISA flags remain explicit inputs so GCC and Clang experiments
# can be reproduced without editing this script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
CXX="${CXX:-g++-15}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-8}"

if ! command -v "${CXX}" >/dev/null 2>&1; then
  printf 'error: C++ compiler is unavailable: %s\n' "${CXX}" >&2
  exit 1
fi

compiler_name="$(basename -- "${CXX}")"
BUILD_DIR="${BUILD_DIR:-${GENDIL_ROOT}/build-k3-${compiler_name}}"

cmake_args=(
  -S "${GENDIL_ROOT}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_CXX_COMPILER="${CXX}"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DBUILD_TESTING=ON
  -DGENDIL_ENABLE_BENCHMARKS=ON
  -DUSE_OPENMP=ON
  -DUSE_CUDA=OFF
  -DUSE_HIP=OFF
  -DUSE_MFEM=OFF
  -DUSE_HYPRE=OFF
  -DUSE_RAJA=OFF
  -DUSE_CALIPER=OFF
)

if [[ -n "${K3_CXX_FLAGS_RELEASE:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_FLAGS_RELEASE=${K3_CXX_FLAGS_RELEASE}")
fi

# Additional arguments permit SDK paths or a generator without changing the
# recorded workflow, for example: build.sh -G Ninja.
cmake "${cmake_args[@]}" "$@"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}" \
  --target gendil-tests range-benchmarks

printf 'K3 build ready: %s\n' "${BUILD_DIR}"
