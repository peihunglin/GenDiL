# K3 Mass Timing Review: 2026-09-03

## Inputs

Result commit `3e1c7cf` contains two warmup and ten timed applications for the
order-1, 4x4x4 mass case. The policy code under test is commit `2abb1a9`.

## Observed Speedups

| A100 share | GCC 15 speedup | Clang 24 speedup |
|---:|---:|---:|
| 0 | 0.773x | 0.0329x |
| 25 | 2.072x | 0.0402x |
| 50 | 2.435x | 0.0307x |
| 75 | 1.512x | 0.0270x |
| 100 | 0.252x | 0.0070x |

All 10 runs pass placement/VLEN validation and report zero mass error. The
large compiler disagreement and extreme edge-share values are not credible
steady-state kernel conclusions for this tiny workload.

## Measurement Defect Found

At commit `2abb1a9`, `K3HeterogeneousOpenMPConfiguration::BlockLoop` rebound
every worker on every operator call. The timed region therefore included
repeated `sched_setaffinity` and `/proc/set_ai_thread` operations. The order-1
case has only 64 cells, so placement overhead dominates the operator work.

The policy is now changed to bind once per reused OpenMP worker ID. The next
timed sweep must be collected after that change. No current speedup should be
used for optimization or compiler comparison.

## Next Run

```sh
git pull --ff-only
GENDIL_K3_MASS_WARMUP=2 GENDIL_K3_MASS_ITERATIONS=20 \
  scripts/machines/spacemit-k3/run-k3-mass-share-sweep.sh
```

After steady-state timing is corrected, increase the mass test to larger
compile-time orders and meshes. Only then compare compiler performance or tune
the A100 share.
