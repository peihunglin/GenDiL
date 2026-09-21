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
inline thread_local Integer tile_call_count = 0;

inline void ResetTileCallCount()
{
   tile_call_count = 0;
}

inline Integer GetTileCallCount()
{
   return tile_call_count;
}

#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
extern "C" void gendil_k3_ime_macc_8x8x8(
   const Storage * lhs,
   const Storage * rhs_transposed,
   float * accumulator );
#endif

GENDIL_HOST_DEVICE inline void MultiplyAccumulate8x8x8(
   const Storage * lhs,
   const Storage * rhs_transposed,
   float * accumulator )
{
   ++tile_call_count;
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
   gendil_k3_ime_macc_8x8x8( lhs, rhs_transposed, accumulator );
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
