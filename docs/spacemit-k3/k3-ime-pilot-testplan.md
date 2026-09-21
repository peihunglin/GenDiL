# K3 IME Pilot Test Plan

## Objective
Validate `InterpContraction` IME pilot on SpacemiT K3 A100 with FP16 operand
packing, FP32 accumulation, and scalar correctness. The portable experiment
mode emulates the instruction so this baseline can run on non-K3 hosts.

## Modes
- `GENDIL_ENABLE_K3_IME_EXPERIMENTS` does not emit IME instructions. It uses
  the portable FP16-packing/FP32-accumulation baseline unless the separately
  opt-in `GENDIL_ENABLE_K3_IME_NATIVE=ON` is configured on K3 hardware.
- `GENDIL_ENABLE_K3_IME_NATIVE` is blocked on non-RISC-V hosts and must only be
  enabled after the installed toolchain accepts and disassembles
  `smt.vfwmadot`. Native configuration validates and applies
  `-mcpu=spacemit-a100` to all `GENDIL::GENDIL` consumers.
- Offline verification needs a compiler with native `_Float16` storage.
- Native verification needs K3 X100/A100 hardware with `ai`/`aix`, a passing
  placement/VLEN probe, and a toolchain that accepts `smt.vfwmadot`.

## Test Matrix

### 1. Offline FP16 baseline
- Configure `-DGENDIL_ENABLE_K3_IME_EXPERIMENTS=ON` on a host compiler with
  native `_Float16` storage support.
- Run `ctest -R '^ime-pilot-correctness$' --output-on-failure`.
- The test validates 8x8x8 FP16 operand packing, nonzero FP32 accumulation,
  and the IME right-operand layout without emitting a K3 instruction.

### 2. Detection sanity
- Run `tools/spacemit-k3/k3-heterogeneous-openmp-probe` to confirm `vlen_bytes=32` on X100 workers, `vlen_bytes=128` on A100 workers.
- Verify `K3HeterogeneousOpenMPConfiguration::OnA100()` returns true on A100 threads.

### 3. Native IME correctness
- Build `ime-pilot-correctness` with both `GENDIL_ENABLE_K3_IME_EXPERIMENTS`
  and `GENDIL_ENABLE_K3_IME_NATIVE` enabled.
- Start the executable through `ai` and compare all 64 results against an
  independent FP32 packed-tile reference.
- Inspect its disassembly and retain evidence of `smt.vfwmadot`.
- Before an end-to-end interpolation run, verify FP16 mode control, nonzero
  accumulation, and multiple K tiles on A100.
- Build and run `ime-interpolation-correctness` normally on X100 with
  `GENDIL_K3_A100_SHARE=50`. It compares 32 independent 2D tensor
  interpolations with an FP64 reference and requires a positive A100 IME tile
  count.

### 4. Tile size coverage
- ND = 8, 16, 24. Verify the zero-padded K tails where applicable.
- NQ = 8, 16, 32. Verify zero-padded M and flattened-N tails.

### 5. Performance smoke
- Benchmark `mass-3d` with `GENDIL_K3_A100_SHARE=50`.
- Collect DoF/s for X100 vs A100, ensure no regression.

## Artifacts
- `results/k3-ime-pilot/` with logs, CSV, and diffs.
- Commit hash tagged in results.

## Rollback
Undefine `GENDIL_ENABLE_K3_IME_EXPERIMENTS`. Revert to scalar path.

## Independent FP16 Baseline
The independent `GENDIL_ENABLE_K3_FP16_BASELINE=ON` and
`GENDIL_ENABLE_K3_IME_EXPERIMENTS=OFF` build shares the A100 detection
interface and registers `fp16-baseline-correctness`. The current test is an
A100/VLEN smoke test only; add a scalar-reference interpolation oracle before
using it as a numerical or performance baseline. See
[`k3-ime-implementation.md`](k3-ime-implementation.md) for the exact
configuration and commands.

## Precision Scope
The experiment does not change the project-wide `gendil::Real` alias, which
remains FP64. IME packs local basis and DoF tiles to FP16, accumulates each
tile in FP32, and converts only the final tile values back to `Real`. A global
FP16 GenDiL build would alter public APIs, storage, and non-IME numerical
paths; it requires a separate precision-port design and is not a prerequisite
for this IME experiment.
