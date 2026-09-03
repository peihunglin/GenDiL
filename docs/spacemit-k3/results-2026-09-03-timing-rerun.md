# K3 Corrected Timing Sweep Review: 2026-09-03

## Evidence

Result commit `1b167b1` contains the first sweep after worker binding was
changed to occur once per reused worker ID. Each run uses two warmups and ten
timed applications of the order-1, 4x4x4 mass operator.

| Compiler | Shares | Speedup range | Correctness |
|---|---|---:|---|
| GCC 15.2 | 0, 25, 50, 75, 100 | 1.63x to 2.98x | all pass, zero error |
| Clang 24 | 0, 25, 50, 75, 100 | 0.016x to 0.054x | all pass, zero error |

Placement and VLEN validation passes for all 10 runs. The heterogeneous policy
continues to report `vlenb=32` on X100 and `vlenb=128` on A100, with no canary,
class, or VLEN failures.

## Interpretation

The GCC result shows useful throughput for this small case. The Clang result is
not an RVV correctness failure: the output is exact and placement is stable.
It indicates that the current policy's 16-thread region and synchronization
overhead are much more expensive under the Clang OpenMP runtime for only 64
cells. The reference path uses eight threads and the heterogeneous path creates
its own 16-thread region for each operator application.

The current policy is therefore validated functionally but is not yet a fair
steady-state performance backend. A production design should either keep a
persistent team across repeated operator applications or use sufficiently
large work units that team overhead is amortized. Compiler comparisons must use
the same team topology and larger orders.

## Next Experiment

Before adding advection or tuned microkernels:

1. Use the new compile-time order 2 and order 3 mass variants with larger meshes.
2. Repeat shares 0, 25, 50, 75, and 100 under both compilers.
3. Collect 1, 2, 4, 8, and 16-thread host baselines where placement allows.
4. Record separate setup, warmup, and steady-state operator time.
5. Decide whether a persistent OpenMP team API is required from the scaling data.

Do not select an A100 share or claim a Clang regression from the current order-1
timings.
