// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/k3ime.hpp"
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)

namespace
{

using namespace gendil;
using namespace gendil::details::k3;

template < Integer M, Integer N, Integer K >
float RunMatrixCase( const bool seed_accumulator )
{
   std::array< float, M * N > actual{};
   std::array< float, M * N > expected{};
   for ( Integer m = 0; m < M; ++m )
   {
      for ( Integer n = 0; n < N; ++n )
      {
         const float initial = seed_accumulator
            ? 0.125f * ( static_cast< float >( m ) - static_cast< float >( n ) )
            : 0.0f;
         actual[ m * N + n ] = initial;
         expected[ m * N + n ] = initial;
         for ( Integer k = 0; k < K; ++k )
         {
            const Storage lhs = ToStorage( 0.25 + 0.125 * m - 0.0625 * k );
            const Storage rhs = ToStorage( -0.5 + 0.0625 * n + 0.03125 * k );
            expected[ m * N + n ] += static_cast< float >( lhs )
               * static_cast< float >( rhs );
         }
      }
   }

   for ( Integer m0 = 0; m0 < M; m0 += TILE_M )
   {
      for ( Integer n0 = 0; n0 < N; n0 += TILE_N )
      {
         float accumulator[ TILE_M * TILE_N ]{};
         for ( Integer m = 0; m < TILE_M && m0 + m < M; ++m )
         {
            for ( Integer n = 0; n < TILE_N && n0 + n < N; ++n )
               accumulator[ m * TILE_N + n ] = actual[ ( m0 + m ) * N + n0 + n ];
         }

         for ( Integer k0 = 0; k0 < K; k0 += TILE_K )
         {
            Storage lhs[ TILE_M * TILE_K ]{};
            Storage rhs_transposed[ TILE_N * TILE_K ]{};
            for ( Integer m = 0; m < TILE_M && m0 + m < M; ++m )
            {
               for ( Integer k = 0; k < TILE_K && k0 + k < K; ++k )
                  lhs[ m * TILE_K + k ] = ToStorage(
                     0.25 + 0.125 * ( m0 + m ) - 0.0625 * ( k0 + k ) );
            }
            for ( Integer n = 0; n < TILE_N && n0 + n < N; ++n )
            {
               for ( Integer k = 0; k < TILE_K && k0 + k < K; ++k )
                  rhs_transposed[ n * TILE_K + k ] = ToStorage(
                     -0.5 + 0.0625 * ( n0 + n ) + 0.03125 * ( k0 + k ) );
            }
            MultiplyAccumulate8x8x8( lhs, rhs_transposed, accumulator );
         }

         for ( Integer m = 0; m < TILE_M && m0 + m < M; ++m )
         {
            for ( Integer n = 0; n < TILE_N && n0 + n < N; ++n )
               actual[ ( m0 + m ) * N + n0 + n ] = accumulator[ m * TILE_N + n ];
         }
      }
   }

   float max_error = 0.0f;
   for ( Integer index = 0; index < M * N; ++index )
      max_error = std::max( max_error, std::abs( actual[ index ] - expected[ index ] ) );
   return max_error;
}

constexpr std::array< const char *, 6 > case_names = {
   "full 8x8x8",
   "seeded 8x8x8",
   "K tail 8x8x9",
   "M tail 9x8x8",
   "N tail 8x9x8",
   "combined tails 9x10x9" };

std::array< float, case_names.size() > RunSuite()
{
   return {
      RunMatrixCase< 8, 8, 8 >( false ),
      RunMatrixCase< 8, 8, 8 >( true ),
      RunMatrixCase< 8, 8, 9 >( false ),
      RunMatrixCase< 9, 8, 8 >( false ),
      RunMatrixCase< 8, 9, 8 >( false ),
      RunMatrixCase< 9, 10, 9 >( true ) };
}

bool ReportSuite( const std::array< float, case_names.size() > & errors )
{
   constexpr float tolerance = 1.e-5f;
   bool passed = true;
   for ( Integer index = 0; index < errors.size(); ++index )
   {
      std::cout << case_names[ index ] << " max error=" << errors[ index ] << '\n';
      passed = passed && errors[ index ] <= tolerance;
   }
   return passed;
}

} // namespace

int main()
{
   static_assert( sizeof( Storage ) == 2, "The IME experiment requires two-byte FP16 storage." );

#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
   constexpr Integer work_items = 32;
   std::array< std::array< float, case_names.size() >, work_items > worker_errors{};
   std::array< int, work_items > ran_on_a100{};
   K3HeterogeneousOpenMPConfiguration::BlockLoop(
      work_items,
      [&] ( const GlobalIndex index )
      {
         if ( K3HeterogeneousOpenMPConfiguration::OnA100() )
         {
            ran_on_a100[ index ] = 1;
            worker_errors[ index ] = RunSuite();
         }
      } );

   Integer a100_items = 0;
   std::array< float, case_names.size() > max_errors{};
   for ( Integer index = 0; index < work_items; ++index )
   {
      a100_items += ran_on_a100[ index ];
      for ( Integer test = 0; test < case_names.size(); ++test )
         max_errors[ test ] = std::max( max_errors[ test ], worker_errors[ index ][ test ] );
   }
   std::cout << "A100 IME matrix suites=" << a100_items << '\n';
   return a100_items > 0 && ReportSuite( max_errors ) ? 0 : 1;
#else
   return ReportSuite( RunSuite() ) ? 0 : 1;
#endif
}

#endif
