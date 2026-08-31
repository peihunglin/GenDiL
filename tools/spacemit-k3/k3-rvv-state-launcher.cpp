// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>

#if !defined( PR_RISCV_V_SET_CONTROL )
#error "Linux headers do not provide RISC-V vector-state control"
#endif

int main( int argc, char ** argv )
{
   if ( argc < 2 )
   {
      std::fprintf( stderr, "usage: %s PROGRAM [ARGUMENT ...]\n", argv[ 0 ] );
      return 2;
   }

   // NEXT occupies bits 3:2. RVV is disabled across exec and inherited by the
   // OpenMP workers until each worker explicitly enables it after placement.
   constexpr unsigned long next_off =
      PR_RISCV_V_VSTATE_CTRL_OFF << 2;
   constexpr unsigned long control =
      next_off | PR_RISCV_V_VSTATE_CTRL_INHERIT;
   if ( prctl( PR_RISCV_V_SET_CONTROL, control ) != 0 )
   {
      std::fprintf(
         stderr,
         "PR_RISCV_V_SET_CONTROL failed: %s\n",
         std::strerror( errno ) );
      return 3;
   }

   execv( argv[ 1 ], &argv[ 1 ] );
   std::fprintf( stderr, "execv(%s) failed: %s\n", argv[ 1 ], std::strerror( errno ) );
   return 4;
}
