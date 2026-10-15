// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

#include <cstdint>
#include <cstring>

namespace gendil {

#if GENDIL_LOW_PRECISION_REAL

namespace internal {

/**
 * @brief Atomic addition on a 16-bit floating-point value.
 *
 * Neither CUDA/HIP nor OpenMP expose atomic addition for `_Float16` or
 * `__bf16`. The value is updated with a compare-and-swap loop instead:
 * on device the loop operates on the 32-bit word containing the half,
 * on host on the 16-bit object itself through the GCC/Clang atomic builtins.
 *
 * @return The value stored before the addition.
 */
GENDIL_HOST_DEVICE inline
Real LowPrecisionAtomicAdd( Real * address, const Real val )
{
   static_assert( sizeof( Real ) == 2, "Low precision atomic add expects a 16-bit type." );
#ifdef GENDIL_DEVICE_CODE
   const std::uintptr_t raw_address = reinterpret_cast< std::uintptr_t >( address );
   unsigned int * word_address = reinterpret_cast< unsigned int * >( raw_address & ~std::uintptr_t{ 2 } );
   const unsigned int shift = ( raw_address & 2 ) ? 16u : 0u;
   const unsigned int mask = 0xffffu << shift;

   unsigned int old_word = *word_address;
   unsigned int assumed_word;
   do
   {
      assumed_word = old_word;
      const unsigned short old_bits = static_cast< unsigned short >( ( assumed_word & mask ) >> shift );
      Real old_value;
      __builtin_memcpy( &old_value, &old_bits, sizeof( Real ) );
      const Real new_value = old_value + val;
      unsigned short new_bits;
      __builtin_memcpy( &new_bits, &new_value, sizeof( Real ) );
      const unsigned int new_word = ( assumed_word & ~mask ) | ( static_cast< unsigned int >( new_bits ) << shift );
      old_word = atomicCAS( word_address, assumed_word, new_word );
   }
   while ( assumed_word != old_word );

   const unsigned short result_bits = static_cast< unsigned short >( ( old_word & mask ) >> shift );
   Real result;
   __builtin_memcpy( &result, &result_bits, sizeof( Real ) );
   return result;
#else
   unsigned short * bits_address = reinterpret_cast< unsigned short * >( address );
   unsigned short expected_bits = __atomic_load_n( bits_address, __ATOMIC_RELAXED );
   Real old_value;
   do
   {
      std::memcpy( &old_value, &expected_bits, sizeof( Real ) );
      const Real new_value = old_value + val;
      unsigned short new_bits;
      std::memcpy( &new_bits, &new_value, sizeof( Real ) );
      if ( __atomic_compare_exchange_n(
              bits_address, &expected_bits, new_bits, true,
              __ATOMIC_RELAXED, __ATOMIC_RELAXED ) )
      {
         break;
      }
   }
   while ( true );
   return old_value;
#endif
}

}

GENDIL_HOST_DEVICE inline Real AtomicAdd( Real & add, const Real val )
{
   return internal::LowPrecisionAtomicAdd( &add, val );
}

GENDIL_HOST_DEVICE inline void AtomicAddInPlace( Real & add, const Real val )
{
   (void) internal::LowPrecisionAtomicAdd( &add, val );
}

#endif // GENDIL_LOW_PRECISION_REAL

template <typename T>
GENDIL_HOST_DEVICE T AtomicAdd(T &add, const T val)
{
#ifdef GENDIL_DEVICE_CODE
   return atomicAdd(&add,val);
#else
   T old;
   #pragma omp atomic capture
   {
      old = add;
      add += val;
   }
   return old;
#endif
}

template <typename T>
GENDIL_HOST_DEVICE void AtomicAddInPlace(T &add, const T val)
{
#ifdef GENDIL_DEVICE_CODE
   (void) atomicAdd(&add,val);
#else
   #pragma omp atomic update
   add += val;
#endif
}

}
