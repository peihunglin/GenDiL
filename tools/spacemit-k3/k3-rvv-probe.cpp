// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "k3-rvv-probe.hpp"

extern "C" int gendil_k3_probe_rvv(
   std::size_t * vlen_bytes,
   std::size_t * vlmax_e64m1,
   const std::uint64_t seed )
{
   constexpr std::size_t max_k3_e64m1_lanes = 16;
   constexpr double increment = 3.0;
   alignas( 128 ) double input[ max_k3_e64m1_lanes ];
   alignas( 128 ) double output[ max_k3_e64m1_lanes ];
   for ( std::size_t lane = 0; lane < max_k3_e64m1_lanes; ++lane )
   {
      input[ lane ] = static_cast< double >( ( seed & 1023ULL ) + lane );
      output[ lane ] = 0.0;
   }

   std::size_t bytes = 0;
   std::size_t vlmax = 0;
   const std::size_t requested_lanes = max_k3_e64m1_lanes;

   // The requested lane count is at least VLMAX on both K3 core classes. The
   // load/add/store canary exercises every physical e64/m1 lane and validates
   // that vector context remains usable after OpenMP synchronization.
   asm volatile(
      "csrr %0, vlenb\n\t"
      "vsetvli %1, %2, e64, m1, ta, ma\n\t"
      "vle64.v v8, (%3)\n\t"
      "vfadd.vf v8, v8, %4\n\t"
      "vse64.v v8, (%5)"
      : "=&r"( bytes ), "=&r"( vlmax )
      : "r"( requested_lanes ), "r"( input ), "f"( increment ), "r"( output )
      : "v8", "memory" );

   *vlen_bytes = bytes;
   *vlmax_e64m1 = vlmax;
   if ( vlmax > max_k3_e64m1_lanes )
   {
      return 0;
   }
   for ( std::size_t lane = 0; lane < vlmax; ++lane )
   {
      if ( output[ lane ] != input[ lane ] + increment )
      {
         return 0;
      }
   }
   return 1;
}
