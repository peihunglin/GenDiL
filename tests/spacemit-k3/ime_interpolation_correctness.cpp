// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/k3ime.hpp"
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

template < typename Actual, typename Expected, size_t ... Is >
Real MaxDifferenceImpl(
   const Actual & actual,
   const Expected & expected,
   std::index_sequence< Is ... > )
{
   Real max_error = 0.0;
   Loop< get_tensor_size_v< Is, Actual > ... >(
      [&] ( auto ... indices )
      {
         max_error = std::max(
            max_error,
            std::abs( actual( indices ... ) - expected( indices ... ) ) );
      } );
   return max_error;
}

template < typename Actual, typename Expected >
Real MaxDifference( const Actual & actual, const Expected & expected )
{
   static_assert( get_rank_v< Actual > == get_rank_v< Expected > );
   return MaxDifferenceImpl(
      actual,
      expected,
      std::make_index_sequence< get_rank_v< Actual > >{} );
}

struct WorkResult
{
   Real max_error = 0.0;
   Integer ime_tiles = 0;
   bool on_a100 = false;
};

template < Integer Dofs, Integer Quads >
bool RunValueCase( const char * name )
{
   using ShapeFunctions = GaussLegendreShapeFunctions< Dofs - 1 >;
   using Points = GaussLegendrePoints< Quads >;
   using Map = CachedDofToQuad< ShapeFunctions, Points >;
   using DofTensor = SerialRecursiveArray< Real, Dofs, Dofs >;

   constexpr Integer work_items = 32;
   constexpr Real tolerance = 1.e-2;
   const auto quad_data = MakeTensorProductData( Map{}, Map{} );
   DofTensor dofs{};
   Loop< Dofs, Dofs >(
      [&] ( const Integer i, const Integer j )
      {
         dofs( i, j ) = 0.25 * static_cast< Real >( i + 1 )
            - 0.125 * static_cast< Real >( j + 1 )
            + 0.03125 * static_cast< Real >( i * j );
      } );

   // Outside BlockLoop no worker has A100 state, so this is the FP64 path.
   const auto reference = InterpolateValuesSerial( quad_data, dofs );
   std::array< WorkResult, work_items > results{};

   K3HeterogeneousOpenMPConfiguration::BlockLoop(
      work_items,
      [&] ( const GlobalIndex index )
      {
         details::k3::ResetTileCallCount();
         const auto actual = InterpolateValuesSerial( quad_data, dofs );
         results[ index ] = WorkResult{
            MaxDifference( actual, reference ),
            details::k3::GetTileCallCount(),
            K3HeterogeneousOpenMPConfiguration::OnA100() };
      } );

   Real max_error = 0.0;
   Integer a100_items = 0;
   Integer a100_ime_tiles = 0;
   for ( const auto & result : results )
   {
      max_error = std::max( max_error, result.max_error );
      if ( result.on_a100 )
      {
         ++a100_items;
         a100_ime_tiles += result.ime_tiles;
      }
   }

   std::cout << name << ": max error=" << max_error
             << " A100 work items=" << a100_items
             << " A100 IME tiles=" << a100_ime_tiles << '\n';
   return a100_items > 0 && a100_ime_tiles > 0 && max_error <= tolerance;
}

} // namespace

int main()
{
#if !defined(GENDIL_ENABLE_K3_IME_NATIVE)
   std::cerr << "Native K3 IME is disabled\n";
   return 2;
#else
   const bool full_tile_passed = RunValueCase< 8, 8 >( "2D 8x8 tile" );
   const bool tail_tile_passed = RunValueCase< 9, 10 >( "2D 9x10 tail" );
   if ( !full_tile_passed || !tail_tile_passed )
   {
      std::cerr << "K3 IME tensor interpolation validation failed\n";
      return 1;
   }
   return 0;
#endif
}
