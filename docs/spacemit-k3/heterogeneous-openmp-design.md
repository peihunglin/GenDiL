# Heterogeneous X100/A100 OpenMP Design

## Purpose

The target execution model is one process with 16 OpenMP workers: eight fixed
to X100 and eight fixed to A100. X100 has 256-bit RVV registers and A100 has
1024-bit RVV registers. A worker must never move between core classes after it
has established vector state.

This document defines the safety experiment that must pass before a mixed-core
GenDiL kernel is implemented. Failure at any gate selects the two-rank MPI
fallback instead of weakening the checks.

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

## Vector-State Hazard

`ai`/`aix` safely moves a new process to A100 before `exec`, but cannot split an
existing process into X100 and A100 worker groups. Moving an OpenMP worker after
the runtime or a library has used RVV can restore vector state with the wrong
VLEN.

Linux exposes per-thread RISC-V vector controls through
`PR_RISCV_V_SET_CONTROL`. The experiment uses them as follows:

1. A scalar launcher requests vector state `OFF` for the next `exec` and marks
   that `NEXT=OFF` policy persistent across later `exec` calls. Worker threads
   are expected to inherit the current disabled state through thread creation,
   and the probe verifies that expectation independently with `GET_CONTROL`.
2. The probe, dynamic loader, C++ runtime, and initial OpenMP worker startup run
   with RVV disabled. An unexpected vector instruction terminates the probe,
   which is a safety failure rather than something to bypass.
3. X100 workers pin themselves to CPUs 0-7.
4. A100 workers write their Linux thread IDs to `/proc/set_ai_thread` while RVV
   remains disabled.
5. Each worker enables RVV for itself only after placement.
6. Workers execute scalable full-lane FP64 load/add/store canaries, read
   `vlenb`, and repeatedly cross OpenMP barriers while checking their CPU
   class, VLEN, and arithmetic output.

The coordinator and placement source are compiled for `rv64gc`. The RVV probe
is isolated in a translation unit compiled for scalable `rv64gcv`. This keeps
vector instructions out of the code that runs before worker placement.

## Go/No-Go Criteria

The single-process design may proceed only when GCC 15 and Clang 24 both show:

- Exactly 16 OpenMP workers.
- Workers 0-7 pinned to X100 CPUs 0-7.
- Workers 8-15 successfully moved to A100 CPUs 8-15 by their individual TIDs.
- `vlenb == 32` and e64/m1 VLMAX of 4 on every X100 worker.
- `vlenb == 128` and e64/m1 VLMAX of 16 on every A100 worker.
- No worker changes core class or VLEN over repeated barriers.
- The process exits normally with all checks passing while vector state was
  disabled during loader and OpenMP startup.

Passing is necessary but not sufficient. The production implementation must
retain a scalar launch path, explicit worker placement, fixed team size, and
runtime assertions in debug builds.

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

## MPI Fallback

If per-thread movement or vector-state control fails, use two processes:

- X100 rank launched normally with eight OpenMP workers.
- A100 rank launched through `ai` before `MPI_Init`, also with eight workers.
- `MPI_THREAD_FUNNELED`; MPI calls occur outside parallel regions.
- Immutable input is duplicated or shared read-only.
- Each rank initially writes a private partial output, followed by an MPI
  reduction. Independent OpenMP runtimes must not coordinate writes through
  OpenMP atomics.

This costs memory and reduction bandwidth but preserves the non-migration
invariant and provides a correct reference implementation.

## Verification And Rollback

The probe is opt-in through `GENDIL_ENABLE_K3_EXPERIMENTS` or the K3 script's
`K3_ENABLE_EXPERIMENTS=ON`; it is not installed, registered with CTest, or
included in ordinary builds. Interprocedural optimization is disabled for the
probe targets so vector instructions cannot cross the scalar/RVV translation
unit boundary. Remove the option and `tools/spacemit-k3` subdirectory to roll
back the experiment without affecting GenDiL headers or execution policies.
