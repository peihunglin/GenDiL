# K3 IME Pilot Test Plan

## Objective
Validate `InterpContraction` IME pilot on SpacemiT K3 A100 with mixed X100/A100 OpenMP run, uniform FP16, scalar correctness.

## Prerequisites
- K3 X100/A100 hardware with `ai`/`aix` tooling.
- Build with `-DGENDIL_ENABLE_K3_IME_EXPERIMENTS=ON -DUSE_OPENMP=ON`.
- Branch `k3-ime-pilot-interp`.

## Test Matrix

### 1. Detection sanity
- Run `tools/spacemit-k3/k3-heterogeneous-openmp-probe` to confirm `vlen_bytes=32` on X100 workers, `vlen_bytes=128` on A100 workers.
- Verify `K3HeterogeneousOpenMPConfiguration::OnA100()` returns true on A100 threads.

### 2. Correctness
- Unit test: `tests/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevalues.cpp`
- Run with `ctest -R interpolate` under mixed OpenMP (`OMP_NUM_THREADS=16`).
- Compare output with scalar reference, tolerance `1e-3` for FP16.

### 3. Tile size coverage
- ND = 8, 16, 24. Ensure remainder path falls back to scalar.
- NQ = 8, 16, 32. Block loop over quadrature points in 8-tile steps.

### 4. Performance smoke
- Benchmark `mass-3d` with `GENDIL_K3_A100_SHARE=50`.
- Collect DoF/s for X100 vs A100, ensure no regression.

## Artifacts
- `results/k3-ime-pilot/` with logs, CSV, and diffs.
- Commit hash tagged in results.

## Rollback
Undefine `GENDIL_ENABLE_K3_IME_EXPERIMENTS`. Revert to scalar path.
