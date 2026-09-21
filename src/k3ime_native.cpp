// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

extern "C" __attribute__((noinline)) void gendil_k3_ime_macc_8x8x8(
   const _Float16 * lhs,
   const _Float16 * rhs_transposed,
   float * accumulator )
{
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
}
