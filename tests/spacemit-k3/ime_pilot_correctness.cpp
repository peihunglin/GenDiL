#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/k3ime.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)

int main()
{
   using namespace gendil;
   using namespace gendil::details::k3;

   static_assert( sizeof( Storage ) == 2, "The offline IME baseline requires compiler FP16 storage." );

   Storage lhs[ TILE_M * TILE_K ];
   Storage rhs_transposed[ TILE_N * TILE_K ];
   float actual[ TILE_M * TILE_N ];
   float expected[ TILE_M * TILE_N ];

   for ( Integer m = 0; m < TILE_M; ++m )
   {
      for ( Integer k = 0; k < TILE_K; ++k )
         lhs[ m * TILE_K + k ] = ToStorage( 0.25 + 0.125 * m - 0.0625 * k );
   }
   for ( Integer n = 0; n < TILE_N; ++n )
   {
      for ( Integer k = 0; k < TILE_K; ++k )
         rhs_transposed[ n * TILE_K + k ] = ToStorage( -0.5 + 0.0625 * n + 0.03125 * k );
   }
   for ( Integer m = 0; m < TILE_M; ++m )
   {
      for ( Integer n = 0; n < TILE_N; ++n )
      {
         actual[ m * TILE_N + n ] = expected[ m * TILE_N + n ] = 0.125f * ( m - n );
         for ( Integer k = 0; k < TILE_K; ++k )
            expected[ m * TILE_N + n ] += static_cast<float>( lhs[ m * TILE_K + k ] )
               * static_cast<float>( rhs_transposed[ n * TILE_K + k ] );
      }
   }

   MultiplyAccumulate8x8x8( lhs, rhs_transposed, actual );
   for ( Integer index = 0; index < TILE_M * TILE_N; ++index )
      assert( std::abs( actual[ index ] - expected[ index ] ) < 1.e-6f );

   std::cout << "IME FP16 packing/FP32 accumulation baseline passed" << std::endl;
   return 0;
}

#else

int main()
{
   std::cout << "IME experiments disabled" << std::endl;
   return 0;
}

#endif
