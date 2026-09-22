// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace gendil;

namespace
{

Integer ReadPositiveEnvironment( const char * name, const Integer fallback )
{
   const char *value = std::getenv( name );
   if ( value == nullptr || value[ 0 ] == '\0' )
      return fallback;
   const auto parsed = std::strtoull( value, nullptr, 10 );
   if ( parsed == 0 )
   {
      std::cerr << name << " must be positive\n";
      std::exit( 2 );
   }
   return static_cast< Integer >( parsed );
}

const char * BenchmarkMode()
{
#if defined(GENDIL_ENABLE_K3_IME_NATIVE)
   return "ime-native";
#elif defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
   return "ime-emulation";
#elif defined(GENDIL_ENABLE_K3_FP16_BASELINE)
   return "fp16-fp32-scalar";
#else
   return "fp64-scalar";
#endif
}

template < Integer NumPoints >
struct TensorTestPoints
{
   static constexpr Integer GetNumPoints() { return NumPoints; }
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

template < Integer Quads, typename Tensor >
Real SumValues( const Tensor & values )
{
   Real sum = 0.0;
   Loop< Quads, Quads >(
      [&] ( const Integer q0, const Integer q1 ) { sum += values( q0, q1 ); } );
   return sum;
}

template < Integer Dofs, Integer Quads >
void RunCase(
   const Integer work_items,
   const Integer warmup,
   const Integer iterations )
{
   using ShapeFunctions = TensorTestShapeFunctions< Dofs >;
   using Points = TensorTestPoints< Quads >;
   using Map = CachedDofToQuad< ShapeFunctions, Points >;
   using DofTensor = SerialRecursiveArray< Real, Dofs, Dofs >;

   const auto quad_data = MakeTensorProductData( Map{}, Map{} );
   DofTensor dofs{};
   Loop< Dofs, Dofs >(
      [&] ( const Integer i, const Integer j )
      {
         dofs( i, j ) = 0.25 * static_cast< Real >( i + 1 )
            - 0.125 * static_cast< Real >( j + 1 )
            + 0.03125 * static_cast< Real >( i * j );
      } );
   std::vector< Real > checksums( work_items );

   const auto RunBatch = [&] ( const Integer count )
   {
      K3HeterogeneousOpenMPConfiguration::BlockLoop(
         work_items,
         [&] ( const GlobalIndex item )
         {
            Real checksum = 0.0;
            for ( Integer iteration = 0; iteration < count; ++iteration )
            {
               const auto values = InterpolateValuesSerial( quad_data, dofs );
               checksum += SumValues< Quads >( values );
            }
            checksums[ item ] = checksum;
         } );
   };

   RunBatch( warmup );
   const auto start = std::chrono::steady_clock::now();
   RunBatch( iterations );
   const auto finish = std::chrono::steady_clock::now();
   const double seconds = std::chrono::duration< double >( finish - start ).count();
   Real checksum = 0.0;
   for ( const Real value : checksums )
      checksum += value;

   const double applications = static_cast< double >( work_items * iterations );
   std::cout << std::setprecision( 17 )
             << "k3-ime-interpolation-benchmark"
             << " mode=" << BenchmarkMode()
             << " dofs_per_dim=" << Dofs
             << " quads_per_dim=" << Quads
             << " work_items=" << work_items
             << " warmup=" << warmup
             << " iterations=" << iterations
             << " seconds=" << seconds
             << " interpolations_per_second=" << applications / seconds
             << " input_dofs_per_second=" << applications * Dofs * Dofs / seconds
             << " output_values_per_second=" << applications * Quads * Quads / seconds
             << " checksum=" << checksum << '\n';
}

} // namespace

int main()
{
   const Integer work_items = ReadPositiveEnvironment( "GENDIL_K3_IME_WORK_ITEMS", 32768 );
   const Integer warmup = ReadPositiveEnvironment( "GENDIL_K3_IME_WARMUP", 5 );
   const Integer iterations = ReadPositiveEnvironment( "GENDIL_K3_IME_ITERATIONS", 1000 );
   RunCase< 8, 8 >( work_items, warmup, iterations );
   RunCase< 9, 10 >( work_items, warmup, iterations );
   return 0;
}
