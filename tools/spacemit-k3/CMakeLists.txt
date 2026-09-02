if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv64|riscv)")
  message(
    FATAL_ERROR
    "GENDIL_ENABLE_K3_EXPERIMENTS requires native RISC-V Linux"
  )
endif()

if(NOT USE_OPENMP)
  message(
    FATAL_ERROR
    "GENDIL_ENABLE_K3_EXPERIMENTS requires USE_OPENMP=ON"
  )
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=rv64gc" GENDIL_K3_HAS_SCALAR_ARCH_FLAG)
check_cxx_compiler_flag("-march=rv64gcv" GENDIL_K3_HAS_RVV_ARCH_FLAG)
if(NOT GENDIL_K3_HAS_SCALAR_ARCH_FLAG OR NOT GENDIL_K3_HAS_RVV_ARCH_FLAG)
  message(
    FATAL_ERROR
    "K3 probes require compiler support for -march=rv64gc and -march=rv64gcv"
  )
endif()

add_executable(
  k3-heterogeneous-openmp-probe
  k3-heterogeneous-openmp-probe.cpp
  k3-rvv-probe.cpp
)
target_compile_features(k3-heterogeneous-openmp-probe PRIVATE cxx_std_20)
target_compile_options(k3-heterogeneous-openmp-probe PRIVATE -fno-lto)
target_link_options(k3-heterogeneous-openmp-probe PRIVATE -fno-lto)
target_link_libraries(
  k3-heterogeneous-openmp-probe
  PRIVATE OpenMP::OpenMP_CXX
)
set_source_files_properties(
  k3-heterogeneous-openmp-probe.cpp
  PROPERTIES COMPILE_OPTIONS "-march=rv64gc"
)
set_source_files_properties(
  k3-rvv-probe.cpp
  PROPERTIES COMPILE_OPTIONS "-march=rv64gcv"
)

add_executable(
  k3-heterogeneous-openmp-mass
  k3-heterogeneous-openmp-mass.cpp
)
target_compile_features(k3-heterogeneous-openmp-mass PRIVATE cxx_std_20)
target_compile_definitions(
  k3-heterogeneous-openmp-mass
  PRIVATE GENDIL_K3_HETERO_OPENMP
)
target_compile_options(k3-heterogeneous-openmp-mass PRIVATE -fno-lto -march=rv64gcv)
target_link_options(k3-heterogeneous-openmp-mass PRIVATE -fno-lto)
target_link_libraries(
  k3-heterogeneous-openmp-mass
  PRIVATE GENDIL::GENDIL OpenMP::OpenMP_CXX
)

add_custom_target(k3-experiments)
add_dependencies(
  k3-experiments
  k3-heterogeneous-openmp-probe
  k3-heterogeneous-openmp-mass
)
