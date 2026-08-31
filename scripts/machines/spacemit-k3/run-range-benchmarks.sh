#!/usr/bin/env bash

# Run every benchmark in benchmarks/range-benchmarks.txt and preserve each
# program's native output. This checks completion and finite output; numerical
# correctness is established separately by CTest until benchmark-level
# reference checks are added.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
BUILD_DIR="${1:?usage: run-range-benchmarks.sh BUILD_DIR OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: run-range-benchmarks.sh BUILD_DIR OUTPUT_DIR}"
CORE_TYPE="${CORE_TYPE:-x100}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OMP_PLACES="${OMP_PLACES:-cores}"
OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
OMP_STACKSIZE="${OMP_STACKSIZE:-64M}"
K3_STACK_SIZE_KB="${K3_STACK_SIZE_KB:-65536}"
BENCHMARK_MODE="${BENCHMARK_MODE:-full}"

if [[ "${BENCHMARK_MODE}" == "smoke" ]]; then
  GENDIL_BENCHMARK_MAX_DOFS="${GENDIL_BENCHMARK_MAX_DOFS:-2000000}"
  GENDIL_BENCHMARK_ITERATIONS="${GENDIL_BENCHMARK_ITERATIONS:-1}"
elif [[ "${BENCHMARK_MODE}" != "full" ]]; then
  printf 'error: BENCHMARK_MODE must be smoke or full\n' >&2
  exit 1
fi

export CORE_TYPE OMP_NUM_THREADS OMP_PLACES OMP_PROC_BIND OMP_STACKSIZE
export BENCHMARK_MODE
export GENDIL_BENCHMARK_MAX_DOFS GENDIL_BENCHMARK_ITERATIONS

if [[ "${CORE_TYPE}" != "x100" && "${CORE_TYPE}" != "a100" ]]; then
  printf 'error: CORE_TYPE must be x100 or a100\n' >&2
  exit 1
fi

if [[ "${K3_STACK_SIZE_KB}" != "0" ]]; then
  ulimit -s "${K3_STACK_SIZE_KB}"
fi

if [[ "${CORE_TYPE}" == "a100" && "${GENDIL_K3_A100_PROCESS:-0}" != "1" ]]; then
  if ! command -v ai >/dev/null 2>&1; then
    printf 'error: CORE_TYPE=a100 requires the k3_ai ai launcher\n' >&2
    exit 1
  fi
  exec ai env GENDIL_K3_A100_PROCESS=1 CORE_TYPE="${CORE_TYPE}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" OMP_PLACES="${OMP_PLACES}" \
    OMP_PROC_BIND="${OMP_PROC_BIND}" OMP_STACKSIZE="${OMP_STACKSIZE}" \
    K3_STACK_SIZE_KB="${K3_STACK_SIZE_KB}" \
    BENCHMARK_MODE="${BENCHMARK_MODE}" \
    GENDIL_BENCHMARK_MAX_DOFS="${GENDIL_BENCHMARK_MAX_DOFS:-}" \
    GENDIL_BENCHMARK_ITERATIONS="${GENDIL_BENCHMARK_ITERATIONS:-}" \
    bash "$0" "$@"
fi

mkdir -p "${OUTPUT_DIR}"
MANIFEST="${OUTPUT_DIR}/manifest.txt"
SUMMARY="${OUTPUT_DIR}/range-benchmarks.csv"
TARGETS_FILE="${GENDIL_ROOT}/benchmarks/range-benchmarks.txt"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'commit=%s\n' "$(git -C "${GENDIL_ROOT}" rev-parse HEAD)"
  printf 'build_dir=%s\n' "${BUILD_DIR}"
  printf 'core_type=%s\n' "${CORE_TYPE}"
  printf 'a100_launcher=%s\n' "${GENDIL_K3_A100_PROCESS:-0}"
  printf 'omp_num_threads=%s\n' "${OMP_NUM_THREADS}"
  printf 'omp_places=%s\n' "${OMP_PLACES}"
  printf 'omp_proc_bind=%s\n' "${OMP_PROC_BIND}"
  printf 'omp_stacksize=%s\n' "${OMP_STACKSIZE}"
  printf 'process_stack_kb=%s\n' "$(ulimit -s)"
  printf 'benchmark_mode=%s\n' "${BENCHMARK_MODE}"
  printf 'benchmark_max_dofs=%s\n' "${GENDIL_BENCHMARK_MAX_DOFS:-full}"
  printf 'benchmark_iterations=%s\n' "${GENDIL_BENCHMARK_ITERATIONS:-default}"
  uname -a
  cmake -LA -N "${BUILD_DIR}" | \
    awk '/^(CMAKE_BUILD_RPATH|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS|CMAKE_CXX_FLAGS_RELEASE|OpenMP_.*_LIBRARY|USE_OPENMP):/'
} > "${MANIFEST}"

printf 'target,status,elapsed_seconds,coordinate_count,log\n' > "${SUMMARY}"
failures=0

while IFS= read -r target; do
  [[ -z "${target}" ]] && continue
  executable="${BUILD_DIR}/benchmarks/${target}"
  log_file="${OUTPUT_DIR}/${target}.txt"
  status="pass"

  if [[ ! -x "${executable}" ]]; then
    printf 'missing executable: %s\n' "${executable}" > "${log_file}"
    status="missing"
    elapsed=0
    coordinate_count=0
  else
    printf 'Running %s on %s with %s OpenMP threads\n' \
      "${target}" "${CORE_TYPE}" "${OMP_NUM_THREADS}"
    start_seconds="${SECONDS}"
    set +e
    if [[ -n "${RUN_TIMEOUT:-}" ]] && command -v timeout >/dev/null 2>&1; then
      timeout "${RUN_TIMEOUT}" "${executable}" > "${log_file}" 2>&1
    else
      "${executable}" > "${log_file}" 2>&1
    fi
    exit_code=$?
    set -e
    elapsed=$((SECONDS - start_seconds))

    coordinate_count="$({
      grep -Eo '\([[:space:]]*[0-9]+,[[:space:]]*[0-9.eE+-]+\)' \
        "${log_file}" || true
    } | wc -l | tr -d '[:space:]')"

    if [[ ${exit_code} -ne 0 ]]; then
      status="exit-${exit_code}"
    elif [[ "${coordinate_count}" == "0" ]]; then
      status="no-output"
    elif grep -Eiq '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "${log_file}"; then
      status="nonfinite"
    fi
  fi

  if [[ "${status}" != "pass" ]]; then
    failures=$((failures + 1))
  fi
  printf '%s,%s,%s,%s,%s\n' \
    "${target}" "${status}" "${elapsed}" "${coordinate_count}" \
    "$(basename -- "${log_file}")" >> "${SUMMARY}"
done < "${TARGETS_FILE}"

if [[ ${failures} -ne 0 ]]; then
  printf '%s range benchmark(s) failed; see %s\n' "${failures}" "${SUMMARY}" >&2
  exit 1
fi

printf 'All range benchmarks completed; results: %s\n' "${OUTPUT_DIR}"
