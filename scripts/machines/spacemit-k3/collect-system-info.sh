#!/usr/bin/env bash

# Collect the compiler, ISA, topology, and OpenMP facts needed before selecting
# K3 build flags. This script intentionally probes capabilities instead of
# encoding unverified vendor flags.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
OUTPUT_FILE="${1:-${GENDIL_ROOT}/k3-system-info.txt}"
OUTPUT_DIR="$(dirname -- "${OUTPUT_FILE}")"
mkdir -p "${OUTPUT_DIR}"

exec > >(tee "${OUTPUT_FILE}") 2>&1

section()
{
  printf '\n===== %s =====\n' "$1"
}

run_optional()
{
  local command_name="$1"
  shift
  if command -v "${command_name}" >/dev/null 2>&1; then
    "$@"
  else
    printf '%s is not available\n' "${command_name}"
  fi
}

section "Survey metadata"
printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'repository=%s\n' "${GENDIL_ROOT}"
git -C "${GENDIL_ROOT}" rev-parse HEAD
git -C "${GENDIL_ROOT}" status --short --branch

section "Kernel and architecture"
uname -a
run_optional lscpu lscpu
run_optional getconf getconf LONG_BIT
run_optional getconf getconf _NPROCESSORS_ONLN

section "RISC-V CPU information"
if [[ -r /proc/cpuinfo ]]; then
  awk '/^(processor|hart|isa|uarch|model name|cpu family|vendor_id)[[:space:]]*:/' /proc/cpuinfo
else
  printf '/proc/cpuinfo is not readable\n'
fi

section "A100 launcher"
if command -v ai >/dev/null 2>&1; then
  printf 'ai=%s\n' "$(command -v ai)"
else
  printf 'ai is not available; install https://github.com/brucehoult/k3_ai\n'
fi
if [[ -e /proc/set_ai_thread ]]; then
  ls -l /proc/set_ai_thread
else
  printf '/proc/set_ai_thread is not present\n'
fi

section "Build tools"
run_optional cmake cmake --version
run_optional ninja ninja --version
run_optional make make --version

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT
cat > "${TEMP_DIR}/openmp-probe.cpp" <<'EOF'
#include <iostream>
#include <omp.h>

int main()
{
  int threads = 0;
#pragma omp parallel reduction(+ : threads)
  threads += 1;
  std::cout << "openmp=" << _OPENMP << " threads=" << threads << '\n';
  return 0;
}
EOF

SEEN_COMPILERS=":"
read -r -a COMPILER_CANDIDATES <<< "${CXX_CANDIDATES:-g++-15 clang++-24 g++ clang++}"
for candidate in "${COMPILER_CANDIDATES[@]}"; do
  if ! command -v "${candidate}" >/dev/null 2>&1; then
    continue
  fi

  compiler_path="$(command -v "${candidate}")"
  if [[ "${SEEN_COMPILERS}" == *":${compiler_path}:"* ]]; then
    continue
  fi
  SEEN_COMPILERS="${SEEN_COMPILERS}${compiler_path}:"

  section "Compiler ${compiler_path}"
  "${compiler_path}" --version
  "${compiler_path}" -dumpmachine || true

  printf '%s\n' '-- Native target expansion --'
  "${compiler_path}" -### -march=native -x c++ -c /dev/null \
    -o "${TEMP_DIR}/native.o" || true

  printf '%s\n' '-- Relevant predefined macros --'
  "${compiler_path}" -dM -E -x c++ /dev/null | \
    awk '/__riscv|__GNUC__|__clang__|__VERSION__|__STDCPP/' || true

  probe_name="$(basename -- "${compiler_path}")"
  probe_path="${TEMP_DIR}/openmp-${probe_name}"
  printf '%s\n' '-- OpenMP probe --'
  if "${compiler_path}" -std=c++20 -O2 -fopenmp \
      "${TEMP_DIR}/openmp-probe.cpp" -o "${probe_path}"; then
    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close "${probe_path}"
    if command -v ai >/dev/null 2>&1; then
      OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close ai "${probe_path}"
    fi
  else
    printf 'OpenMP probe compilation failed for %s\n' "${compiler_path}"
  fi
done

section "Relevant environment"
for name in CC CXX CFLAGS CXXFLAGS LDFLAGS OMP_NUM_THREADS OMP_PLACES \
  OMP_PROC_BIND LD_LIBRARY_PATH PATH; do
  printf '%s=%s\n' "${name}" "${!name-}"
done

printf '\nSurvey written to %s\n' "${OUTPUT_FILE}"
