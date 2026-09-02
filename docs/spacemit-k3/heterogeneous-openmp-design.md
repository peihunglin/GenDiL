# Heterogeneous X100/A100 OpenMP Design

## Purpose

The target execution model is one process with 16 OpenMP workers: eight fixed
to X100 and eight fixed to A100. X100 has 256-bit RVV registers and A100 has
1024-bit RVV registers. A worker must never move between core classes after it
has established vector state.

This document defines the empirical OpenMP experiment used before a mixed-core
GenDiL kernel is implemented. It follows the working K3 GEMM reference path;
it is a K3-specific runtime contract, not a portable OpenMP guarantee.

## Why One Scalable Kernel Comes First

RVV code sets `vl` dynamically. A vector-length-agnostic FP64 kernel can
therefore execute on both 256-bit and 1024-bit implementations without two
instruction variants. Core-specific variants are useful only when measurements
justify different LMUL, unrolling, tiling, or prefetch choices.

GenDiL currently defines `Real` as `double`, including matrix-free temporary
storage and coefficients. The first heterogeneous backend will consequently be
FP64. General FP32 support requires a separate scalar-type design; silently
converting existing FP64 vectors inside selected kernels would not constitute
general GenDiL FP32 support.

## K3 OpenMP Placement

`ai`/`aix` remains required for a process dedicated to A100, but the mixed
OpenMP process starts normally on X100. Inside one persistent OpenMP team:

1. Workers 0-7 pin themselves to X100 CPUs 0-7 with `sched_setaffinity`.
2. Workers 8-15 write their Linux TIDs to `/proc/set_ai_thread`.
3. A barrier separates placement from the first explicit RVV canary.
4. Workers execute scalable FP64 load/add/store canaries and repeatedly check
   CPU class, VLEN, affinity, and arithmetic output.

This intentionally matches `rvv-evaluation/proxy_bench/gemm/gemm_hetero.c`.
The runtime establishes vector state during ordinary startup, so the earlier
vector-disabled `prctl` probe is retained only as a failed historical safety
experiment and is not the active execution contract.

The reference builds the X100 and A100 worker objects separately with
`-mcpu=spacemit-x100` and `-mcpu=spacemit-a100`, respectively. GenDiL will use
the same object-level separation once the probe passes; a single compiler
translation unit must not be expected to optimize both microarchitectures
equally.

## Go/No-Go Criteria

The single-process design may proceed only when GCC 15 and Clang 24 both show:

- Exactly 16 OpenMP workers.
- Workers 0-7 pinned to X100 CPUs 0-7.
- Workers 8-15 successfully confined to the A100 CPU class by their individual
  TIDs.
- `vlenb == 32` and e64/m1 VLMAX of 4 on every X100 worker.
- `vlenb == 128` and e64/m1 VLMAX of 16 on every A100 worker.
- No worker changes core class or VLEN over repeated barriers.
- The process exits normally with all checks passing after worker placement.

Passing is necessary but not sufficient. The production implementation must
retain a scalar launch path, explicit worker placement, fixed team size, and
runtime assertions in debug builds.

These criteria passed for GCC 15 and Clang 24 in result commit `bf5149e` using
probe commit `b18adc5`. The A100 criterion is class confinement, not singleton
CPU affinity.

## Planned Work Dispatch

Standard OpenMP tasks will not select an architecture: task affinity is only a
hint and tasks can execute on another worker. A heterogeneous policy will own a
fixed parallel region and manually assign ranges:

```text
workers 0..7   -> X100 range -> scalable or X100-tuned FP64 kernel
workers 8..15  -> A100 range -> scalable or A100-tuned FP64 kernel
```

The X100/A100 boundary will be weighted using measured throughput for the
operator, dimension, order, and precision. Each eight-worker group then divides
its contiguous range statically. The existing `BlockLoop(count, body)` API must
gain an explicit range/heterogeneous execution path; nested OpenMP regions are
not acceptable.

One-process OpenMP atomics remain valid for shared DoF accumulation, but their
contention and numerical ordering require validation. Face and H1 operators
will follow disjoint DG cell operators because disjoint cell ranges do not
imply disjoint algebraic output ranges.

## Historical Safety-Gate Result

The September 2 vector-disabled probe produced `SIGILL` under both GCC and
Clang before program output. This rejects only that stricter startup experiment;
it does not reject the empirically demonstrated K3 GEMM-style placement path.
Keep the result in `docs/spacemit-k3/results-2026-09-02.md` for traceability.

## Verification And Rollback

The probe is opt-in through `GENDIL_ENABLE_K3_EXPERIMENTS` or the K3 script's
`K3_ENABLE_EXPERIMENTS=ON`; it is not installed, registered with CTest, or
included in ordinary builds. The probe compiles its coordinator and RVV canary
in separate translation units with LTO disabled. Remove the option and
`tools/spacemit-k3` subdirectory to roll back the experiment without affecting
GenDiL headers or execution policies.
