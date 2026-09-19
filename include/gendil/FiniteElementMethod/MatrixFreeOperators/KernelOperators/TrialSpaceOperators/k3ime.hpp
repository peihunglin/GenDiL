// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
#endif

namespace gendil::details::k3
{

// IME consumes FP16 operands but GenDiL's public scalar remains Real/FP64.
#if defined(__FLT16_MANT_DIG__)
using Storage = _Float16;
#else
using Storage = float;
#endif

inline Storage ToStorage( Real v )
{
   return static_cast< Storage >( v );
}

inline bool IsA100()
{
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
   return K3HeterogeneousOpenMPConfiguration::OnA100();
#else
   // Portable instruction emulation permits offline packing and accuracy tests.
   return true;
#endif
}

constexpr Integer TILE_M = 8;
constexpr Integer TILE_N = 8;
constexpr Integer TILE_K = 8;

GENDIL_HOST_DEVICE
inline void MultiplyAccumulate8x8x8(
   const Storage * lhs,
   const Storage * rhs_transposed,
   float * accumulator )
{
#if defined(GENDIL_ENABLE_K3_IME_NATIVE) && defined(__riscv)
   // A100 Xsmtfp16fp32mm: C += A * B^T. rhs_transposed contains one
   // contiguous K-vector per output column, as required by the IME layout.
   asm volatile(
      "vsetvli t0, zero, e16, m1\n\t"
      "vle16.v v2, (%[lhs])\n\t"
      "vle16.v v8, (%[rhs])\n\t"
      "vsetvli t0, zero, e32, m2\n\t"
      "vle32.v v16, (%[acc])\n\t"
      "vsetvli t0, zero, e16, m1\n\t"
      "smt.vfwmadot v16, v2, v8\n\t"
      "vsetvli t0, zero, e32, m2\n\t"
      "vse32.v v16, (%[acc])\n\t"
      :
      : [lhs] "r"( lhs ), [rhs] "r"( rhs_transposed ), [acc] "r"( accumulator )
      : "memory", "t0" );
#else
   for ( Integer m = 0; m < TILE_M; ++m )
   {
      for ( Integer n = 0; n < TILE_N; ++n )
      {
         float value = accumulator[ m * TILE_N + n ];
         for ( Integer k = 0; k < TILE_K; ++k )
         {
            value += static_cast< float >( lhs[ m * TILE_K + k ] )
               * static_cast< float >( rhs_transposed[ n * TILE_K + k ] );
         }
         accumulator[ m * TILE_N + n ] = value;
      }
   }
#endif
}

} // namespace gendil::details::k3
