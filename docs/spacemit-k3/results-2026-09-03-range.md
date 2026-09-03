# K3 Mixed Range Sweep Review: 2026-09-03

## Evidence

Result commit `718c0ed` contains mixed-core range runs from code commit
`e252fd5`:

- GCC 15 and Clang 24.
- A100 shares 25, 50, and 75 percent.
- 18 range targets per share and compiler.
- Performance mode, 2,000,000-item cap, seven timed iterations.
- 16 OpenMP workers, eight nominal X100 workers and eight nominal A100
  workers.

All 108 target runs pass the runner's execution checks:

- No missing executable.
- No nonzero process exit.
- Nonempty coordinate output.
- No detected `NaN` or `Inf` output.
- Expected coordinate counts for every target/share/compiler combination.

## Interpretation

The K3 policy successfully executes all 18 range programs across both compiler
families and all tested mixed shares. This confirms that the cell-index range
partition does not deadlock or crash the current volume, face, p-adaptive, or
h-adaptive benchmark paths at the configured performance cap.

This is not a numerical correctness result. The range programs print PGFPlots
throughput coordinates but do not emit checksums or compare against a reference
operator. Shared H1/global-face writes and face accumulation can be wrong while
still producing finite timing output. CTest was not run through the
heterogeneous benchmark policy in this sweep.

The CSV `elapsed_seconds` field is an integer wall-clock duration for the whole
executable and is useful for collection diagnostics only. Use the per-target
coordinate throughput values for performance analysis after adding correctness
oracles and repeating with larger stable work units.

## Next Gates

1. Add reference-value or mixed-vs-host comparison checks to representative
   range benchmarks, starting with mass/advection 3D and 6D.
2. Run those correctness checks at shares 0, 25, 50, 75, and 100.
3. Add larger caps/orders after correctness passes.
4. Parse coordinate curves into compiler/share comparison tables rather than
   comparing coarse integer process durations.
5. Only then tune the A100 share or introduce separately compiled X100/A100
   RVV microkernels.

## Rollback

The mixed range behavior is enabled only by `GENDIL_ENABLE_K3_EXPERIMENTS=ON`.
Disable that option to restore the existing host `SerialKernelConfiguration`
for all range benchmarks. The benchmark source files and their original output
format remain unchanged.
