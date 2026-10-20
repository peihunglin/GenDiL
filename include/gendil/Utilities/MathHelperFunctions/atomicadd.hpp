// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

#include <atomic>

namespace gendil {

#if GENDIL_LOW_PRECISION_REAL

namespace internal {

/**
 * @brief Atomic addition on a 16-bit floating-point value.
 *
 * OpenMP does not expose atomic addition for `_Float16` or `__bf16`. The host
 * implementation updates the 16-bit object itself through C++20 atomic_ref.
 * Low precision device builds are rejected at configure time.
 *
 * @return The value stored before the addition.
 */
GENDIL_HOST_DEVICE inline
Real LowPrecisionAtomicAdd( Real * address, const Real val )
{
   static_assert( sizeof( Real ) == 2, "Low precision atomic add expects a 16-bit type." );
   static_assert(
      std::atomic_ref< Real >::required_alignment <= alignof( Real ),
      "The platform requires over-aligned low-precision atomic storage." );
#if defined( GENDIL_DEVICE_CODE )
#error "Low-precision device atomics are not supported. Configure FP16/BF16 with USE_CUDA=OFF and USE_HIP=OFF."
#else
   std::atomic_ref< Real > atomic_value( *address );
   Real expected = atomic_value.load( std::memory_order_relaxed );
   Real old_value;
   do
   {
      old_value = expected;
      const Real new_value = old_value + val;
      if ( atomic_value.compare_exchange_weak(
              expected, new_value,
              std::memory_order_relaxed, std::memory_order_relaxed ) )
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
