# Range Benchmark Harness

## Rationale

The original range programs hard-coded their sweep limits and iteration counts.
Several allocate for up to 100 million DoFs, and the 6D speed-of-light program
allows one billion. Running every program at those limits is inappropriate for
an initial compiler or runtime check and can obscure a porting failure behind
compile time or memory pressure.

The K3 workflow also requires a stable list of all range programs. A broad
`benchmarks` target is insufficient because it builds unrelated executables and
does not prove that every required range target was included.

## Design

`benchmarks/range-benchmarks.txt` lists the 18 required executable targets.
CMake creates `range-benchmarks` from that list. A non-CUDA configuration fails
if any listed target does not exist. CUDA retains its existing exclusions and
is not a K3 target configuration.

`benchmarks/include/range-benchmark-config.hpp` reads two optional positive
integer environment variables:

- `GENDIL_BENCHMARK_MAX_DOFS` overrides each program's existing sweep cap.
- `GENDIL_BENCHMARK_ITERATIONS` overrides its existing timed iteration count.

Unset variables preserve the original constants. The controls affect workload
size only; they do not select kernels or alter numerical types.

The K3 runner provides two policies:

- `BENCHMARK_MODE=smoke`: cap at 2,000,000 and use one timed iteration.
- `BENCHMARK_MODE=full`: leave both variables unset and preserve the original
  program behavior.

Explicit environment values take precedence in smoke mode. Every selected
value is copied into `manifest.txt`.

## A100 Process Safety

When `CORE_TYPE=a100`, the test and benchmark runners replace themselves with a
new process launched by `ai`. The marker `GENDIL_K3_A100_PROCESS=1` prevents a
relaunch loop. All CTest and benchmark children inherit A100 placement. This is
preferred to moving a process that may already have established X100 RVV state.

## Verification

Static shell syntax and ShellCheck validation were run when introducing the
scripts. Compilation and runtime verification are intentionally deferred to
the native K3 machine. Required commands are:

```sh
CXX=g++-15 scripts/machines/spacemit-k3/build.sh
BENCHMARK_MODE=smoke CORE_TYPE=x100 \
  scripts/machines/spacemit-k3/run-range-benchmarks.sh \
  build-k3-g++-15 results/spacemit-k3/gcc15/x100-smoke
```

Repeat for Clang 24 and for A100. A full run follows only after all smoke rows
report `pass`.

CTest output defaults to
`results/spacemit-k3/<compiler>/<core>/tests.txt`, preventing one compiler or
core run from overwriting another. An explicit second argument to
`run-tests.sh` overrides that location.

The runner checks process exit status and requires finite coordinate output.
It does not prove numerical correctness. CTest is the current numerical gate;
benchmark-level reference comparisons remain required before RVV or IME
results can be accepted.

## Limitations

- The scripts assume a single-config CMake build, where executables are under
  `<build>/benchmarks/`.
- The workload cap has similar but not identical meaning in uniform,
  p-adaptive, and h-adaptive programs.
- One iteration is suitable only for smoke testing, not performance reporting.
- Full benchmark logs retain the programs' existing PGFPlots-oriented format.

## Rollback

Remove the shared header includes and restore the original constants to remove
runtime workload controls. Remove the manifest loop at the end of
`benchmarks/CMakeLists.txt` to remove the aggregate target. These changes do not
alter installed GenDiL headers or library behavior.
