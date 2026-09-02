#!/usr/bin/env bash

# Validate the safe fallback topology: one X100 MPI rank and one A100 MPI rank,
# each owning an eight-thread OpenMP team. The A100 rank enters through aix
# before its loader, MPI runtime, or OpenMP runtime can establish RVV state.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
BUILD_DIR="${1:?usage: run-mpi-openmp-probe.sh BUILD_DIR [OUTPUT_FILE]}"
PROBE="${BUILD_DIR}/tools/spacemit-k3/k3-mpi-openmp-probe"
compiler_path="$(
  awk -F= '/^CMAKE_CXX_COMPILER:[^=]*=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)"
compiler_name="$(basename -- "${compiler_path:-unknown-compiler}")"
OUTPUT_FILE="${2:-${GENDIL_ROOT}/results/spacemit-k3/${compiler_name}/mpi-openmp-probe-$(git -C "${GENDIL_ROOT}" rev-parse --short HEAD).txt}"
MPIEXEC="${MPIEXEC:-$(
  awk -F= '/^MPIEXEC_EXECUTABLE:[^=]*=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)}"
MPIEXEC_NUMPROC_FLAG="$(
  awk -F= '/^MPIEXEC_NUMPROC_FLAG:[^=]*=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)"
MPIEXEC_PREFLAGS="$(
  awk -F= '/^MPIEXEC_PREFLAGS:[^=]*=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)"
MPIEXEC_POSTFLAGS="$(
  awk -F= '/^MPIEXEC_POSTFLAGS:[^=]*=/{print $2}' \
    "${BUILD_DIR}/CMakeCache.txt"
)"
AIX="${AIX:-$(command -v aix || true)}"
RUN_TIMEOUT="${RUN_TIMEOUT:-120}"

if [[ ! -x "${PROBE}" ]]; then
  printf 'error: MPI fallback probe was not built; verify MPI CXX availability\n' >&2
  exit 1
fi
if [[ -z "${MPIEXEC}" || ! -x "${MPIEXEC}" ]]; then
  printf 'error: MPI launcher is unavailable: %s\n' "${MPIEXEC}" >&2
  exit 1
fi
if [[ -z "${MPIEXEC_NUMPROC_FLAG}" ]]; then
  printf 'error: CMake did not record MPIEXEC_NUMPROC_FLAG\n' >&2
  exit 1
fi
if [[ -n "${MPIEXEC_PREFLAGS}" || -n "${MPIEXEC_POSTFLAGS}" ]]; then
  printf 'error: nonempty CMake MPI pre/post flags require explicit review\n' >&2
  exit 1
fi
if [[ -z "${AIX}" || ! -x "${AIX}" ]]; then
  printf 'error: aix is required to start the A100 MPI rank safely\n' >&2
  exit 1
fi
if ! command -v taskset >/dev/null 2>&1; then
  printf 'error: taskset is required to constrain the X100 MPI rank\n' >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  printf 'error: timeout is required to bound MPI probe failures\n' >&2
  exit 1
fi

mpi_version="$("${MPIEXEC}" --version 2>&1)"
mpi_binding_args=()
if [[ "${mpi_version}" == *"Open MPI"* ||
      "${mpi_version}" == *"OpenRTE"* ]]; then
  mpi_binding_args=(--bind-to none)
elif [[ "${mpi_version}" == *"HYDRA"* ||
       "${mpi_version}" == *"MPICH"* ]]; then
  mpi_binding_args=(-bind-to none)
else
  printf 'error: unsupported MPI launcher; review its MPMD syntax first\n' >&2
  printf '%s\n' "${mpi_version}" >&2
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
  printf 'mpiexec=%s\n' "${MPIEXEC}"
  printf '%s\n' "${mpi_version}"
  printf 'mpi_binding_args=%s\n' "${mpi_binding_args[*]}"
  printf 'aix=%s\n' "${AIX}"
  uname -a
  printf '%s\n' 'probe_output_begin'
  set +e
  env -u OMP_THREAD_LIMIT -u GOMP_CPU_AFFINITY -u KMP_AFFINITY \
    OMP_NUM_THREADS=8 \
    OMP_DYNAMIC=FALSE \
    OMP_PLACES=cores \
    OMP_PROC_BIND=spread \
    timeout --kill-after=10 "${RUN_TIMEOUT}" \
    "${MPIEXEC}" \
      "${mpi_binding_args[@]}" \
      "${MPIEXEC_NUMPROC_FLAG}" 1 taskset -c 0-7 "${PROBE}" x100 \
      : "${MPIEXEC_NUMPROC_FLAG}" 1 "${AIX}" "${PROBE}" a100
  probe_status=$?
  printf '%s\n' 'probe_output_end'
  exit "${probe_status}"
} 2>&1 | tee "${OUTPUT_FILE}"
pipeline_status=("${PIPESTATUS[@]}")
set -e

status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
if [[ ${tee_status} -ne 0 ]]; then
  printf 'failed to write MPI probe output: %s\n' "${OUTPUT_FILE}" >&2
  exit "${tee_status}"
fi
if [[ ${status} -ne 0 ]]; then
  printf 'MPI/OpenMP fallback probe failed; see %s\n' "${OUTPUT_FILE}" >&2
  exit "${status}"
fi

printf 'MPI/OpenMP fallback probe passed; result: %s\n' "${OUTPUT_FILE}"
