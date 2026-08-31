# Repository Guidance

## Build And Test
- GenDiL is a header-only C++20 library; the public umbrella header is `include/gendil/gendil.hpp`.
- Configure out of source: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`.
- OpenMP is required by default. Use `-DUSE_OPENMP=OFF` only for a deliberately serial build.
- Build and run all configured tests with `cmake --build build --parallel` followed by `ctest --test-dir build --output-on-failure`.
- Run one test with `ctest --test-dir build -R '^<exact-test-name>$' --output-on-failure`; CTest names match targets declared under `tests/`.
- Benchmarks are excluded unless configured with `-DGENDIL_ENABLE_BENCHMARKS=ON`.
- MFEM and HYPRE integration tests exist only when `USE_MFEM` or `USE_HYPRE` is enabled and a matching installation is supplied.

## Execution Model
- `SerialKernelConfiguration` aliases `HostKernelConfiguration`; its outer work loop uses OpenMP and is not serial in an OpenMP build.
- CUDA and HIP are mutually exclusive. Enabling either recompiles test, example, and benchmark sources in that device language.
- Release flags default to `-march=native`; override them explicitly when cross-compiling or producing binaries for a different CPU class.
- Debug builds enable AddressSanitizer through the top-level CMake flags.

## SpacemiT K3 Port
- Read `docs/spacemit-k3/README.md` before changing K3 code or running K3 experiments.
- X100 and A100 have different RVV vector lengths. Start A100 processes through `ai`/`aix` before any RVV state is established; never migrate a running vectorized process between core classes.
- Do not add fixed-VLEN assumptions. RVV kernels must remain vector-length agnostic and be validated on both X100 and A100.
- Do not guess IME intrinsics, data types, or flags. Record the installed SDK/compiler interface first.
- Every porting change requires documentation of rationale, design, verification, limitations, and rollback. Keep commits atomic and include focused test evidence.
