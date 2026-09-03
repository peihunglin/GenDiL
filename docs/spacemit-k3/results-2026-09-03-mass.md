# K3 Heterogeneous Mass Review: 2026-09-03

## Evidence

Result commit `bec3c8d` contains the GCC 15 and Clang 24 runs. Both use code
commit `95cc9f9`, X100/A100 OpenMP placement validation, 16 workers, and an
A100 share of 50 percent.

Both runs report:

- 16/16 workers.
- X100 `vlenb=32`, e64/m1 VLMAX=4 for all eight X100 workers.
- A100 `vlenb=128`, e64/m1 VLMAX=16 for all eight A100 workers.
- Zero class changes, VLEN changes, or canary failures over 64 rounds.
- `K3 heterogeneous mass max error: 0`.

This validates the first K3 execution-policy slice for a small order-1,
4x4x4-cell mass operator at a 50/50 work split. It does not yet validate
large-order behavior, performance, adaptive shares, or separate tuned X100 and
A100 kernel objects.

## Next Measurement

Run the five-share sweep with both compilers:

```sh
git pull --ff-only
scripts/machines/spacemit-k3/run-k3-mass-share-sweep.sh
```

The default shares are `0 25 50 75 100`. Override with
`SHARES="10 30 50 70 90"`, or select one compiler with `RUN_GCC=0` or
`RUN_CLANG=0`. Results are stored under compiler-specific directories and one
file per share.

The sweep is intended to expose edge cases where one worker group receives no
cells, before increasing mesh/order sizes. A performance comparison should
follow only after every share validates numerically.
