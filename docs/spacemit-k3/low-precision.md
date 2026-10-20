# K3 FP16 And BF16 Host Support

## Rationale

GenDiL can select compiler-native `_Float16` or `__bf16` as `gendil::Real` for
host/OpenMP experiments on SpacemiT K3. This validates storage, arithmetic, and
contention behavior before any RVV or IME implementation is introduced. It does
not assume an RVV vector length, IME instruction, compiler flag, or accumulation
format.

## Design

- `GENDIL_REAL_TYPE=FP16` and `BF16` use the compiler extended scalar types.
- `RealMinNormal`, `RealMax`, `RealEpsilonValue`, `Sqrt`, and `Abs` are GenDiL
  APIs. C++20 does not authorize adding overloads or specializations for these
  compiler types to `namespace std`.
- Core kernels initialize, assign, and return `Real` values with `Real{ 0 }`
  and `Real{ 1 }` rather than `double` literals. GCC 15 on K3 diagnoses direct
  `double` to `_Float16` conversion as a narrowing conversion when range
  kernels instantiate it.
- Range benchmark mesh spacings, coefficient vectors, and deterministic input
  fills likewise use `Real` arithmetic so their assignments remain valid under
  the K3 GCC extended-type rules.
- Template calls that select scalar `Real` overloads pass `Real{ 0 }` or
  `Real{ 1 }` explicitly; deduction does not apply the required conversion from
  a `double` literal.
- Low-precision integration tests use explicit `Real` mesh dimensions and
  reference expressions for the same reason; this keeps K3 GCC diagnostics out
  of the validation targets.
- Host low-precision accumulation uses a compare-and-swap loop on the actual
  aligned `Real` object. It never reads a neighboring half-word.
- FP16/BF16 are rejected with CUDA, HIP, MFEM, or HYPRE. Those paths have not
  established a compatible scalar ABI or a safe atomic implementation.

## K3 Verification

First record the installed compiler support and flags; do not invent ISA flags:

```sh
scripts/machines/spacemit-k3/collect-system-info.sh results/spacemit-k3/system-info.txt
```

For each available GCC 15 and Clang 24 compiler, configure, build, and run the
focused scalar tests on X100:

```sh
cmake -S . -B build-k3-fp16-gcc -D CMAKE_BUILD_TYPE=Release -D GENDIL_REAL_TYPE=FP16
cmake --build build-k3-fp16-gcc --parallel --target test-real-type test-atomic-add test-nodal-subspace-projection
ctest --test-dir build-k3-fp16-gcc -R '^(test-real-type|test-atomic-add|test-nodal-subspace-projection)$' --output-on-failure
```

Repeat with `GENDIL_REAL_TYPE=BF16` and with Clang. Then run each focused CTest
command through `ai` in a newly created A100 process. Never move an already
vectorized process between X100 and A100. Run the complete configured CTest
suite only after the focused tests pass, recording every failing test and whether
the failure is a compiler capability rejection, build failure, or numerical
failure.

## Acceptance And Limits

The focused tests must pass with `OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=8` on
both core classes. `test-atomic-add` verifies loss-free concurrent increments
within the exactly representable range: 2048 for FP16 and 256 for BF16.
`test-nodal-subspace-projection` instantiates the host projection path and
verifies its reduced-precision result.

## Range Benchmark Build

After the focused tests pass, the complete range benchmark set can be built for
throughput experiments:

```sh
cmake -S . -B build-k3-fp16-gcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GENDIL_REAL_TYPE=FP16 \
  -D GENDIL_ENABLE_BENCHMARKS=ON
cmake --build build-k3-fp16-gcc --parallel --target range-benchmarks
```

Repeat with `GENDIL_REAL_TYPE=BF16` and with Clang. Run the executables from
`build-k3-fp16-gcc/benchmarks` in separate X100 and freshly `ai`-launched A100
processes. Keep the precision, compiler, core class, OpenMP environment, command
line, and benchmark output with each result.

The range programs are throughput measurements, not numerical oracles. Before
accepting a low-precision performance result, compare the selected operator's
output against a double-precision reference for the same mesh and input, and
record the maximum error and tolerance alongside the throughput.

No K3 hardware results are included in this change. The complete test and
example build is not yet an acceptance gate because remaining direct C++20
`std` math and streaming uses on compiler extended types must be migrated to
GenDiL helpers or explicit `float` conversions. Vectorized low-precision
kernels, IME use, mixed-precision accumulation, device backends, and external
`std` math support remain out of scope. Disable the feature by configuring
`-D GENDIL_REAL_TYPE=DOUBLE`; reverting this change removes the FP16/BF16 host
path without affecting the default build.
