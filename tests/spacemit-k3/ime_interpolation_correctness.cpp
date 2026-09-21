// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/k3ime.hpp"
#endif
#if defined(GENDIL_ENABLE_K3_IME_NATIVE) || defined(GENDIL_ENABLE_K3_FP16_BASELINE)
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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

#if defined(GENDIL_ENABLE_K3_IME_NATIVE) || defined(GENDIL_ENABLE_K3_FP16_BASELINE)
struct WorkResult
{
   Real max_error = 0.0;
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
   Integer ime_tiles = 0;
#endif
   bool on_a100 = false;
};

Integer A100WorkItems( const Integer total )
{
   const char *value = std::getenv( "GENDIL_K3_A100_SHARE" );
   if ( value == nullptr || value[ 0 ] == '\0' )
      return total / 2;
   return total * static_cast< Integer >( std::atoi( value ) ) / 100;
}
#endif

template < Integer NumPoints >
struct TensorTestPoints
{
   static constexpr Integer GetNumPoints()
   {
      return NumPoints;
   }

   static constexpr Real GetCoord( const Integer q )
   {
      return -0.875 + 1.75 * static_cast< Real >( q )
         / static_cast< Real >( NumPoints - 1 );
   }

   static constexpr Real GetWeight( const Integer )
   {
      return 1.0 / static_cast< Real >( NumPoints );
   }
};

template < Integer NumDofs >
struct TensorTestShapeFunctions
{
   static constexpr Integer num_dofs = NumDofs;

   static constexpr Real ComputeValue( const Integer dof, const Real point )
   {
      return 0.125 * ( static_cast< Real >( dof % 5 ) - 2.0 )
         + 0.03125 * static_cast< Real >( dof + 1 ) * point;
   }

   static constexpr Real ComputeGradientValue( const Integer dof, const Real )
   {
      return 0.03125 * static_cast< Real >( dof + 1 );
   }
};

template < Integer Dofs, Integer Quads >
bool RunValueCase( const char * name )
{
   // The production Gauss-Legendre table currently ends at eight points.
   // These deterministic maps permit a 9x10 tail case without extending that
   // unrelated numerical-integration table.
   using ShapeFunctions = TensorTestShapeFunctions< Dofs >;
   using Points = TensorTestPoints< Quads >;
   using Map = CachedDofToQuad< ShapeFunctions, Points >;
   using DofTensor = SerialRecursiveArray< Real, Dofs, Dofs >;

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

   SerialRecursiveArray< Real, Quads, Quads > reference{};
   const auto & first_map = GetTensorProductEntry< 0 >( quad_data );
   const auto & second_map = GetTensorProductEntry< 1 >( quad_data );
   Loop< Quads, Quads >(
      [&] ( const Integer q0, const Integer q1 )
      {
         Real value = 0.0;
         for ( Integer d0 = 0; d0 < Dofs; ++d0 )
         {
            for ( Integer d1 = 0; d1 < Dofs; ++d1 )
            {
               value += first_map.values( q0, d0 )
                  * second_map.values( q1, d1 ) * dofs( d0, d1 );
            }
         }
         reference( q0, q1 ) = value;
      } );

#if defined(GENDIL_ENABLE_K3_IME_NATIVE) || defined(GENDIL_ENABLE_K3_FP16_BASELINE)
   constexpr Integer work_items = 32;
   std::array< WorkResult, work_items > results{};

   K3HeterogeneousOpenMPConfiguration::BlockLoop(
      work_items,
      [&] ( const GlobalIndex index )
      {
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
         details::k3::ResetTileCallCount();
#endif
         const auto actual = InterpolateValuesSerial( quad_data, dofs );
         results[ index ] = WorkResult{
            MaxDifference( actual, reference ),
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
            details::k3::GetTileCallCount(),
#endif
            K3HeterogeneousOpenMPConfiguration::OnA100() };
      } );

   Real max_error = 0.0;
   Integer a100_items = 0;
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
   Integer a100_ime_tiles = 0;
#endif
   for ( const auto & result : results )
   {
      max_error = std::max( max_error, result.max_error );
      if ( result.on_a100 )
      {
         ++a100_items;
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
         a100_ime_tiles += result.ime_tiles;
#endif
      }
   }

   const bool expect_a100_work = A100WorkItems( work_items ) > 0;
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
   const bool a100_dispatch_passed = expect_a100_work
      ? a100_items > 0 && a100_ime_tiles > 0
      : a100_items == 0 && a100_ime_tiles == 0;
#else
   const bool a100_dispatch_passed = expect_a100_work
      ? a100_items > 0
      : a100_items == 0;
#endif
   const bool accuracy_passed = max_error <= tolerance;
   std::cout << name << ": max error=" << max_error
              << " A100 work items=" << a100_items
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
              << " A100 IME tiles=" << a100_ime_tiles
#endif
              << " expected A100 work=" << expect_a100_work
              << " accuracy=" << accuracy_passed
              << " dispatch=" << a100_dispatch_passed << '\n';
   return accuracy_passed && a100_dispatch_passed;
#else
   details::k3::ResetTileCallCount();
   const auto actual = InterpolateValuesSerial( quad_data, dofs );
   const Real max_error = MaxDifference( actual, reference );
   const Integer ime_tiles = details::k3::GetTileCallCount();
   std::cout << name << ": offline max error=" << max_error
             << " IME tiles=" << ime_tiles << '\n';
   return max_error <= tolerance && ime_tiles > 0;
#endif
}

} // namespace

int main()
{
   const bool full_tile_passed = RunValueCase< 8, 8 >( "2D 8x8 tile" );
   const bool tail_tile_passed = RunValueCase< 9, 10 >( "2D 9x10 tail" );
   if ( !full_tile_passed || !tail_tile_passed )
   {
      std::cerr << "K3 tensor interpolation validation failed\n";
      return 1;
   }
   return 0;
}
