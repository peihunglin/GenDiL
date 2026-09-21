# K3 IME Interpolation Implementation

## Rationale

The A100 `Xsmtfp16fp32mm` instruction performs an FP16 x FP16 to FP32
8x8x8 matrix multiply-accumulate. A tensor-product interpolation contraction
can be expressed as a batched matrix multiplication, so it is a suitable
experimental IME target. The implementation remains opt-in because GenDiL's
public scalar type is FP64 and the IME conversion changes numerical results.

## Design

`InterpContractionIME` flattens all non-active tensor dimensions into the
matrix N dimension and evaluates:

```text
output[quadrature, batch] += basis[quadrature, dof] * input[dof, batch]
```

For each 8x8x8 tile it:

1. Converts basis and input values from `Real` to `_Float16`.
2. Packs the basis tile in row-major order.
3. Packs one K-vector per output column for the IME right operand.
4. Accumulates the tile in FP32.
5. Converts the completed FP32 result to `Real` when scattering output.

Partial M, N, and K tiles are zero-padded. This preserves the full
contraction result without a separate scalar remainder path.

`GENDIL_ENABLE_K3_IME_EXPERIMENTS=ON` enables the tile interface and a scalar
emulator with identical FP16 packing and FP32 accumulation. It is intended for
offline correctness testing. `GENDIL_ENABLE_K3_IME_NATIVE=ON` additionally
selects the documented `smt.vfwmadot` inline-assembly sequence. Native mode
requires both the IME experiment option and a `riscv64` CMake target. It also
requires compiler support for `-mcpu=spacemit-a100`; CMake validates that flag
and propagates it through the `GENDIL::GENDIL` interface to every consumer of
the native IME headers.

The native path runs only when the K3 execution policy has identified the
current worker as A100. It must never execute on X100.

## Precision Policy

The IME experiment is not a project-wide FP16 build. `gendil::Real` remains
FP64 in all public APIs, matrices, and non-IME kernels. The local IME policy
is:

```text
FP64 input and basis -> FP16 packed operands -> FP32 accumulation -> FP64 output
```

Changing `Real` globally would be a separate scalar-type port. It would affect
public interfaces, storage, assembly, solver tolerances, and all non-IME
kernels; it is neither necessary nor implied by the A100 IME implementation.

## Offline Verification

On a compiler with two-byte `_Float16` storage, configure and run:

```sh
cmake -S . -B build-ime-offline \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENDIL_ENABLE_K3_IME_EXPERIMENTS=ON \
  -DUSE_OPENMP=OFF
cmake --build build-ime-offline --parallel --target ime-pilot-correctness
ctest --test-dir build-ime-offline \
  -R '^ime-pilot-correctness$' --output-on-failure
```

The `ime-pilot-correctness` CTest target validates two-byte FP16 storage,
8x8x8 packing, right-operand orientation, nonzero FP32 initial accumulation,
and the scalar emulator result. It does not execute a K3 instruction.

The initial implementation passed this test with AppleClang 21 on macOS. This
is layout and mixed-precision evidence only, not A100 instruction evidence.

## Native A100 Verification

First run the required placement/VLEN probe normally on X100:

```sh
K3_ENABLE_EXPERIMENTS=ON CXX=/usr/bin/g++-15 \
  scripts/machines/spacemit-k3/build.sh
scripts/machines/spacemit-k3/run-heterogeneous-openmp-probe.sh \
  build-k3-g++-15
```

Then configure the native IME build on K3:

```sh
cmake -S . -B build-k3-ime \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
  -DUSE_OPENMP=ON \
  -DGENDIL_ENABLE_K3_EXPERIMENTS=ON \
  -DGENDIL_ENABLE_K3_IME_EXPERIMENTS=ON \
  -DGENDIL_ENABLE_K3_IME_NATIVE=ON
cmake --build build-k3-ime --parallel --target ime-pilot-correctness
ai build-k3-ime/tests/spacemit-k3/ime-pilot-correctness
```

Inspect the object or executable disassembly and retain evidence that it
contains `smt.vfwmadot`. The first A100 run must also verify FP16 mode control,
the documented result layout, nonzero accumulator behavior, and multiple K
tiles against an independent FP32 reference before enabling end-to-end
interpolation measurements.

The native build also registers `ime-interpolation-correctness`. Run it
normally on X100 so its `BlockLoop` workers perform K3 self-placement:

```sh
cmake --build build-k3-ime --parallel --target ime-interpolation-correctness
GENDIL_K3_A100_SHARE=50 \
  build-k3-ime/tests/spacemit-k3/ime-interpolation-correctness
```

The test evaluates 32 independent 2D tensor-product interpolations, compares
each against an FP64 scalar reference, and requires A100 workers to report a
positive IME tile count. It covers a complete 8x8 tensor case and a 9-DoF,
10-quadrature-point zero-padded tail case. It must run normally, not through
`ai`, because the mixed policy places its own workers.

## FP16 Baseline Without IME

The intended independent baseline configuration is:

```sh
cmake -S . -B build-k3-fp16 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_OPENMP=ON \
  -DGENDIL_ENABLE_K3_EXPERIMENTS=ON \
  -DGENDIL_ENABLE_K3_FP16_BASELINE=ON \
  -DGENDIL_ENABLE_K3_IME_EXPERIMENTS=OFF
```

The A100 detection interface is shared by the IME and baseline experiment
options, so this configuration can be built independently. Build and run its
initial A100 detection smoke test with:

```sh
cmake --build build-k3-fp16 --parallel --target fp16-baseline-correctness
ai build-k3-fp16/tests/spacemit-k3/fp16-baseline-correctness
```

The registered test currently verifies that the baseline configuration can
query A100/VLEN state. It is not yet an end-to-end FP16 interpolation oracle.
Do not treat the IME emulator as a substitute for the independent baseline;
add a scalar-reference interpolation comparison before using either result for
performance acceptance.

## Limitations And Rollback

- Native inline assembly has not been compiled or executed on K3 hardware.
- The installed K3 compiler flags, assembler support, vector-register clobber
  behavior, and FP16 mode-control interface must be recorded on target.
- FP16 conversion and FP32 accumulation introduce error relative to the FP64
  reference; packing, conversion, tail, and correction costs must be included
  in performance results.
- The implementation is only for disjoint cell-local contractions. It does not
  solve shared-DoF or face-output ownership.

Disable the experiment with `-DGENDIL_ENABLE_K3_IME_EXPERIMENTS=OFF`. This
removes the IME dispatch and retains the existing scalar contraction path.
