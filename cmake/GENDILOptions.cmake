include_guard(GLOBAL)

option(USE_MFEM "Enables MFEM interface." OFF)
option(USE_OPENMP "Enables OpenMP parallelization features." ON)
option(USE_CUDA "Enables CUDA support." OFF)
option(USE_HIP "Enables HIP support." OFF)
option(USE_RAJA "Enables RAJA features." OFF)
option(USE_CALIPER "Enables Caliper interface for benchmarking." OFF)
option(USE_HYPRE "Enables Hypre interface." OFF)
option(GENDIL_ENABLE_BENCHMARKS "Build GenDiL benchmark executables" OFF)

set(
  GENDIL_REAL_TYPE
  "DOUBLE"
  CACHE STRING
  "Floating-point type behind gendil::Real: DOUBLE, FLOAT, FP16, or BF16"
)
set_property(CACHE GENDIL_REAL_TYPE PROPERTY STRINGS DOUBLE FLOAT FP16 BF16)
string(TOUPPER "${GENDIL_REAL_TYPE}" GENDIL_REAL_TYPE)
if(NOT GENDIL_REAL_TYPE MATCHES "^(DOUBLE|FLOAT|FP16|BF16)$")
  message(
    FATAL_ERROR
    "GENDIL_REAL_TYPE must be DOUBLE, FLOAT, FP16, or BF16 (received "
    "'${GENDIL_REAL_TYPE}')."
  )
endif()
if(GENDIL_REAL_TYPE MATCHES "^(FP16|BF16)$" AND (USE_MFEM OR USE_HYPRE))
  message(
    FATAL_ERROR
    "GENDIL_REAL_TYPE=${GENDIL_REAL_TYPE} cannot be combined with USE_MFEM "
    "or USE_HYPRE: both interfaces require gendil::Real to be double "
    "precision."
  )
endif()
if(GENDIL_REAL_TYPE MATCHES "^(FP16|BF16)$" AND (USE_CUDA OR USE_HIP))
  message(
    FATAL_ERROR
    "GENDIL_REAL_TYPE=${GENDIL_REAL_TYPE} currently supports host/OpenMP builds "
    "only. Configure with USE_CUDA=OFF and USE_HIP=OFF."
  )
endif()

option(
  GENDIL_ENABLE_FILTERED_SYNC_EXPERIMENT_CTEST
  "Register filtered CellIterator Sync experiment in CTest."
  OFF
)

set(
  GENDIL_DEVICE_SPARSE_FINALIZATION
  "AUTO"
  CACHE STRING
  "GPU sparse finalization support: AUTO, ON, or OFF"
)
set_property(
  CACHE GENDIL_DEVICE_SPARSE_FINALIZATION
  PROPERTY STRINGS AUTO ON OFF
)
option(
  GENDIL_FETCH_ROCPRIM
  "Fetch rocPRIM when HIP sparse finalization cannot find it"
  OFF
)
set(
  GENDIL_ROCPRIM_GIT_TAG
  "rocm-6.3.1"
  CACHE STRING
  "rocPRIM git tag or branch used by GENDIL_FETCH_ROCPRIM"
)

string(
  TOUPPER
  "${GENDIL_DEVICE_SPARSE_FINALIZATION}"
  GENDIL_DEVICE_SPARSE_FINALIZATION
)
if(NOT GENDIL_DEVICE_SPARSE_FINALIZATION MATCHES "^(AUTO|ON|OFF)$")
  message(
    FATAL_ERROR
    "GENDIL_DEVICE_SPARSE_FINALIZATION must be AUTO, ON, or OFF (received "
    "'${GENDIL_DEVICE_SPARSE_FINALIZATION}')."
  )
endif()

set(GENDIL_USE_HYPRE_DEVICE OFF)
set(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION OFF)
set(GENDIL_CUSPARSE_HAS_FLOAT_DOUBLE_SPMV OFF)
