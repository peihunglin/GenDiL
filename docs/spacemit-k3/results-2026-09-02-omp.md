# K3 Empirical OpenMP Probe Review: 2026-09-02

## Inputs

- Probe code: `b18adc55ab8e0c7cf2ce2439a60ddd1a63e02502`.
- Result commit: `bf5149e7d24cdaf934f1730015ef8367cb8dd0f0`.
- Host: SpacemiT K3, Linux 6.18.3, eight X100 and eight A100 cores.
- GCC: Bianbu GCC 15.2.0.
- Clang: LLVM Clang 24.0.0git.
- Process started normally on X100; it was not launched through `ai`.

## Result

Both compiler runs pass:

| Check | GCC 15 | Clang 24 |
|---|---:|---:|
| OpenMP workers | 16/16 | 16/16 |
| X100 workers with `vlenb=32`, e64/m1 VLMAX=4 | 8/8 | 8/8 |
| A100 workers with `vlenb=128`, e64/m1 VLMAX=16 | 8/8 | 8/8 |
| Core-class changes over 64 rounds | 0 | 0 |
| VLEN changes over 64 rounds | 0 | 0 |
| FP64 canary failures | 0 | 0 |
| Probe result | pass | pass |

The probe confirms that the K3 GEMM-style mechanism can maintain one 16-worker
OpenMP process with eight X100 workers and eight A100 workers under both tested
compilers.

## Affinity Limitation

X100 workers have singleton masks for CPUs 0-7. A100 workers report the full
`0xff00` A100-class mask because `/proc/set_ai_thread` selects the A100 class,
not a specific A100 CPU. GCC happened to report distinct final CPUs in this
run; Clang reused some A100 CPUs between workers. This is not a probe failure,
but it means the production policy must not assume one-to-one A100 worker/core
pinning or use `OMP_PLACES=cores` as proof of A100 placement.

## Go Decision

Proceed with the single-process OpenMP design for a K3-specific experimental
GenDiL backend. Do not reintroduce MPI as the default execution path.

The first GenDiL implementation should follow the reference GEMM structure:

1. Create one persistent OpenMP team.
2. Bind X100 workers once with `sched_setaffinity`.
3. Move A100 worker TIDs once through `/proc/set_ai_thread`.
4. Synchronize before entering the first vectorized operator kernel.
5. Compile X100 and A100 kernel translation units separately with their
   respective `-mcpu=spacemit-*` flags.
6. Split a naturally independent work dimension into static X100/A100 ranges.
7. Validate the complete output against the existing FP64 host path.

Start with cell-local volume mass and advection operators. Do not begin with
H1, face, COO, or shared-DoF operators because their output ownership is not
implied by a disjoint cell range.

## Remaining Risks

- The probe validates RVV execution and class stability, not GenDiL's nested
  operator phases or shared-DoF atomics.
- Existing GenDiL `BlockLoop` calls create independent OpenMP regions; a
  persistent heterogeneous team needs a new execution path.
- The probe's RVV object uses scalable `rv64gcv` rather than separate tuned
  X100/A100 kernel objects. Separate object compilation is still required for
  production kernels.
- The result is empirical and K3-specific. It does not establish portability
  to another heterogeneous scheduler or OpenMP runtime.

## Rollback

If a GenDiL mixed operator fails correctness or placement validation, disable
the K3 heterogeneous policy and retain the existing X100-only OpenMP path.
The empirical probe remains a prerequisite for every future mixed-kernel
change.
