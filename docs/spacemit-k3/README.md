# SpacemiT K3 Port

This directory records the design, reproduction steps, and evidence for the
GenDiL K3 port. Code changes must update the relevant document in the same
commit.

Detailed benchmark-harness design and rollback notes are in
[`benchmark-harness.md`](benchmark-harness.md).
The mixed X100/A100 safety design is in
[`heterogeneous-openmp-design.md`](heterogeneous-openmp-design.md).
The first native result review is in
[`results-2026-08-31.md`](results-2026-08-31.md).
The heterogeneous safety-gate result is in
[`results-2026-09-02.md`](results-2026-09-02.md).
The passing empirical OpenMP result is in
[`results-2026-09-02-omp.md`](results-2026-09-02-omp.md).

## Hardware Model

The K3 exposes two core classes:

- Eight out-of-order X100 cores with 256-bit RISC-V vector registers.
- Eight in-order A100 cores with 1024-bit RISC-V vector registers and IME.

Normal Linux processes start on X100. The external `k3_ai` `ai`/`aix` launcher
moves a newly created process to A100 before executing the requested program.
Moving a process after it has established RVV state is unsafe because the two
core classes have different VLENs.

The current production-safe path uses separate process runs for X100 and A100;
the same vector-length-agnostic implementation is validated in a fresh process
on each core class. Concurrent use of all 16 cores is an experimental goal
following the K3 GEMM-style OpenMP placement path. It must pass the empirical
placement/VLEN probe before GenDiL kernels use it.

## Stages

### 1. X100 OpenMP

- Build natively with GCC 15 and Clang 24 using C++20 and OpenMP.
- Run the complete configured CTest suite on one and eight X100 cores.
- Run every `range-*` benchmark with fixed OpenMP affinity.
- Record correctness evidence and 1, 2, 4, and 8-thread scaling.

### 2. X100 And A100 RVV

- Add vector-length-agnostic RVV kernels and aligned storage contracts.
- Prioritize tensor contractions, vector reductions, and sparse matrix rows.
- Build with both compilers and run separate X100 and A100 processes.
- Inspect compiler vectorization reports and disassembly to prove RVV coverage.

### 3. A100 IME

- Record the installed IME headers, compiler flags, supported element types,
  tile shapes, and accumulation semantics before adding an API.
- Add IME behind the same contraction semantics as scalar and RVV paths.
- Compare against FP64 references. If native precision is insufficient,
  evaluate an Ozaki-style split-product scheme with higher-precision
  accumulation and residual correction.
- Include packing, conversion, and correction costs in performance results.

## Initial System Survey

Run from a clean checkout on K3:

```sh
scripts/machines/spacemit-k3/collect-system-info.sh \
  results/spacemit-k3/system-info.txt
```

The script does not assume compiler executable names. Override its candidate
list when needed:

```sh
CXX_CANDIDATES="/path/to/g++ /path/to/clang++" \
  scripts/machines/spacemit-k3/collect-system-info.sh \
  results/spacemit-k3/system-info.txt
```

Install `k3_ai` and ensure `ai` is on `PATH` before the survey. The script runs
its small OpenMP probe normally on X100 and, when `ai` is available, in a fresh
A100 process.

Review generated logs for hostnames, usernames, paths, and environment details
before committing them.

## Range Benchmark Inventory

`benchmarks/range-benchmarks.txt` is the source of truth for K3 performance
coverage. Configuring a non-CUDA build with `-DGENDIL_ENABLE_BENCHMARKS=ON`
creates the aggregate `range-benchmarks` target and fails if any listed
benchmark is unavailable.
Build the complete suite with:

```sh
cmake --build build-k3 --parallel --target range-benchmarks
```

The strict failure is intentional. CUDA currently omits some adaptive range
targets, but all entries must exist in an OpenMP/RVV/IME K3 configuration.

## Native Build And Test

Configure and build the complete test and range suites with either compiler:

```sh
CXX=g++-15 scripts/machines/spacemit-k3/build.sh
CXX=clang++-24 scripts/machines/spacemit-k3/build.sh
```

The default build directories are `build-k3-g++-15` and
`build-k3-clang++-24`. Set `BUILD_DIR`, `JOBS`, or `BUILD_TYPE` to override
them. Set `K3_CXX_FLAGS_RELEASE` only after the system survey establishes the
required ISA flags; its exact value is stored in `CMakeCache.txt` and captured
by the benchmark manifest.

The K3 script defaults release flags to `-O3 -DNDEBUG` because the installed
GCC 15 rejects `-march=native`. Source-specific scalar and RVV flags are used
only by the heterogeneous safety probe. Override `K3_CXX_FLAGS_RELEASE` when a
reviewed common K3 ISA string is introduced.

Run correctness tests on each core class:

```sh
CORE_TYPE=x100 scripts/machines/spacemit-k3/run-tests.sh build-k3-g++-15
CORE_TYPE=a100 scripts/machines/spacemit-k3/run-tests.sh build-k3-g++-15
```

Run all range benchmarks and capture one log per executable:

```sh
BENCHMARK_MODE=smoke CORE_TYPE=x100 \
  scripts/machines/spacemit-k3/run-range-benchmarks.sh \
  build-k3-g++-15 results/spacemit-k3/gcc15/x100-smoke
CORE_TYPE=x100 scripts/machines/spacemit-k3/run-range-benchmarks.sh \
  build-k3-g++-15 results/spacemit-k3/gcc15/x100
CORE_TYPE=a100 scripts/machines/spacemit-k3/run-range-benchmarks.sh \
  build-k3-g++-15 results/spacemit-k3/gcc15/a100
```

Repeat with `build-k3-clang++-24`. The scripts re-execute themselves through
`ai` for A100 before invoking CTest or benchmark binaries. Do not wrap an
already running X100 benchmark process with `/proc/set_ai_thread`.

`BENCHMARK_MODE=smoke` defaults to one timed iteration and a 2,000,000-item
sweep cap. `BENCHMARK_MODE=full` preserves each benchmark's original limits and
iteration count. Either mode can be overridden explicitly with the positive
integer environment variables `GENDIL_BENCHMARK_MAX_DOFS` and
`GENDIL_BENCHMARK_ITERATIONS`; their values are recorded in the manifest.

The range runner rejects missing executables, nonzero exits, empty coordinate
output, and non-finite results. This is an execution sanity check, not a
numerical oracle; CTest remains the correctness gate until benchmark-level
reference checks are implemented.

## Baseline Rerun

After pulling a review fix, run the complete GCC 15 and Clang X100 baseline
with one command:

```sh
git pull --ff-only
scripts/machines/spacemit-k3/rerun-x100-baseline.sh
```

The script uses `/usr/bin/g++-15` and
`/home/lin32/opt/llvm-main/bin/clang++`, rebuilds both configurations, runs
CTest, and runs all 18 smoke benchmarks. It continues after failures so one
compiler cannot prevent collection of the other compiler's results. Output
paths include the tested Git commit by default.

Override installed compiler paths or run only one compiler when needed:

```sh
GCC_CXX=/path/to/g++-15 CLANG_CXX=/path/to/clang++ \
  scripts/machines/spacemit-k3/rerun-x100-baseline.sh
RUN_GCC=0 scripts/machines/spacemit-k3/rerun-x100-baseline.sh
RUN_CLANG=0 scripts/machines/spacemit-k3/rerun-x100-baseline.sh
```

`RESULT_TAG` can replace the default `rerun-<commit>` suffix. Commit all
generated result files even when the script returns nonzero.

After the baseline passes, collect Stage 1 OpenMP scaling data with:

```sh
scripts/machines/spacemit-k3/run-x100-stage1-performance.sh
```

This runs every range benchmark with GCC and Clang at 1, 2, 4, and 8 threads.
Performance mode uses seven timed iterations and the validated smoke-size cap.
Set `RUN_GCC=0`, `RUN_CLANG=0`, `THREAD_COUNTS="1 8"`, or `RESULT_TAG=name` to
control a rerun without editing the script.

## Heterogeneous OpenMP Probe

Before implementing a mixed-core kernel, rebuild and run the vector-state probe
with both compilers:

```sh
git pull --ff-only
K3_ENABLE_EXPERIMENTS=ON CXX=/usr/bin/g++-15 \
  scripts/machines/spacemit-k3/build.sh
scripts/machines/spacemit-k3/run-heterogeneous-openmp-probe.sh \
  build-k3-g++-15

K3_ENABLE_EXPERIMENTS=ON \
  CXX=/home/lin32/opt/llvm-main/bin/clang++ \
  scripts/machines/spacemit-k3/build.sh
scripts/machines/spacemit-k3/run-heterogeneous-openmp-probe.sh \
  build-k3-clang++
```

Run this script normally on X100, never through `ai`. Workers 0-7 pin to X100;
workers 8-15 request A100 placement by Linux TID and only then execute the
explicit RVV canary. A nonzero result blocks mixed-core kernel work; do not
weaken or bypass a failed placement or VLEN check.
K3 experiments are disabled in ordinary builds unless
`K3_ENABLE_EXPERIMENTS=ON` is supplied.

The stricter vector-disabled startup probe failed under both compilers because
runtime startup uses RVV before worker placement. The active probe now follows
the validated K3 GEMM-style OpenMP path: start normally on X100, place selected
worker TIDs on A100, and check VLEN after placement. Do not wrap the mixed probe
with `ai`.

The empirical probe passes under GCC 15 and Clang 24. Proceed with the
single-process OpenMP backend study; do not make MPI a prerequisite.

The next focused target is built with the same opt-in experiment build:

```sh
K3_ENABLE_EXPERIMENTS=ON CXX=/usr/bin/g++-15 \
  scripts/machines/spacemit-k3/build.sh
```

This builds `tools/spacemit-k3/k3-heterogeneous-openmp-mass`, which compares
the K3 policy against the existing host mass operator. Set
`GENDIL_K3_A100_SHARE=50` (or another percentage from 0 to 100) when running
the target. The target must run normally on X100, not through `ai`.

Run the placement check and mass comparison in the required order with:

```sh
GENDIL_K3_A100_SHARE=50 \
  scripts/machines/spacemit-k3/run-k3-mass-study.sh \
  build-k3-g++-15
```

The runner stops before the mass comparison if placement/VLEN validation fails,
and records the share, probe output, and numerical comparison in one result.
The mass comparison performs one warmup and five timed applications by default,
then reports reference and heterogeneous DoF/s and speedup. Override the
protocol with `GENDIL_K3_MASS_WARMUP` and `GENDIL_K3_MASS_ITERATIONS`.

Run both compilers over the initial share matrix with:

```sh
scripts/machines/spacemit-k3/run-k3-mass-share-sweep.sh
```

The default shares are `0 25 50 75 100`; each result is stored under a
compiler-specific directory. See
[`results-2026-09-03-mass.md`](results-2026-09-03-mass.md) for the acceptance
scope.
The completed share-sweep analysis is in
[`results-2026-09-03-share-sweep.md`](results-2026-09-03-share-sweep.md).
The initial timing artifact and required rerun are documented in
[`results-2026-09-03-timing.md`](results-2026-09-03-timing.md).
The corrected timing analysis is in
[`results-2026-09-03-timing-rerun.md`](results-2026-09-03-timing-rerun.md).


## Evidence Required Per Change

Each implementation commit must document:

- Problem and rationale.
- Files, public options, and execution paths changed.
- Compiler and hardware assumptions.
- Exact configure, build, test, and benchmark commands.
- Correctness reference and tolerance.
- Performance result or an explicit statement that performance was not tested.
- Known limitations and how to revert or disable the change.

Use small commits that introduce one abstraction, kernel family, or validation
step. Do not combine infrastructure, RVV, IME, and numerical-policy changes in
one commit.

## Result Layout

Store concise, reviewable evidence as:

```text
results/spacemit-k3/<date>/<compiler>/<core>/
  manifest.txt
  tests.txt
  range-benchmarks.csv
```

The manifest must include the GenDiL commit, compiler version, complete flags,
core class, thread count, affinity variables, OS/kernel version, and whether
`ai` launched the process. Keep bulky temporary compiler and build logs outside
Git unless they are necessary to explain a failure.
