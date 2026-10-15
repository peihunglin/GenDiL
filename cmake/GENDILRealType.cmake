include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

# Configures the floating-point type behind gendil::Real on the given target.
#
# GENDIL_REAL_TYPE selects the type:
#   DOUBLE -> double (default)
#   FLOAT  -> float
#   FP16   -> _Float16 (IEEE binary16)
#   BF16   -> __bf16   (bfloat16)
#
# FP16 and BF16 require compiler support for arithmetic on the corresponding
# extended floating-point type (GCC >= 12 / 13 or Clang >= 15 / 17 on x86-64
# and AArch64). The low precision modes exist to exercise the library and
# its test suite in a purely low precision setting; they are not supported
# together with the MFEM or HYPRE interfaces, which require double precision.
function(gendil_configure_real_type target)
  set(real_definition "GENDIL_REAL_TYPE_${GENDIL_REAL_TYPE}")
  target_compile_definitions(${target} INTERFACE ${real_definition})

  if(GENDIL_REAL_TYPE STREQUAL "FP16")
    set(real_cxx_type "_Float16")
  elseif(GENDIL_REAL_TYPE STREQUAL "BF16")
    set(real_cxx_type "__bf16")
  elseif(GENDIL_REAL_TYPE STREQUAL "FLOAT")
    set(real_cxx_type "float")
  else()
    set(real_cxx_type "double")
  endif()

  if(GENDIL_REAL_TYPE MATCHES "^(FP16|BF16)$")
    # The MFEM/HYPRE conflict is rejected early in GENDILOptions.cmake.
    set(CMAKE_REQUIRED_FLAGS "-std=c++20")
    check_cxx_source_compiles(
      "
      #include <cstring>
      int main()
      {
         ${real_cxx_type} a = static_cast< ${real_cxx_type} >( 1.5f );
         ${real_cxx_type} b = a * a + a - a / a;
         float f = static_cast< float >( b );
         unsigned short bits = 0;
         std::memcpy( &bits, &b, sizeof( bits ) );
         return ( f > 0.0f && bits != 0 ) ? 0 : 1;
      }
      "
      GENDIL_HAS_${GENDIL_REAL_TYPE}_ARITHMETIC
    )
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT GENDIL_HAS_${GENDIL_REAL_TYPE}_ARITHMETIC)
      message(
        FATAL_ERROR
        "GENDIL_REAL_TYPE=${GENDIL_REAL_TYPE} requires a C++ compiler with "
        "arithmetic support for ${real_cxx_type} (GCC >= 13 or Clang >= 17 on "
        "x86-64 or AArch64). The compiler '${CMAKE_CXX_COMPILER}' failed the "
        "probe."
      )
    endif()

    if(USE_CUDA AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
      message(
        FATAL_ERROR
        "GENDIL_REAL_TYPE=${GENDIL_REAL_TYPE} is not supported with NVCC, "
        "which does not implement ${real_cxx_type} arithmetic. Use Clang as "
        "the CUDA compiler instead."
      )
    endif()
  endif()

  message(
    STATUS
    "GENDIL real type: ${GENDIL_REAL_TYPE} (gendil::Real = ${real_cxx_type})"
  )
endfunction()
