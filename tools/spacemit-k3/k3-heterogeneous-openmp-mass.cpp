// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace gendil;

namespace
{

Integer ReadPositiveEnvironment( const char * name, const Integer fallback )
{
   const char *value = std::getenv( name );
   if ( value == nullptr || value[ 0 ] == '\0' )
   {
      return fallback;
   }
   const auto parsed = std::strtoull( value, nullptr, 10 );
   if ( parsed == 0 )
   {
      std::cerr << name << " must be positive\n";
      std::exit( 2 );
   }
   return static_cast< Integer >( parsed );
}

template < typename Operator >
double TimeOperator(
   Operator & op,
   const Vector & input,
   Vector & output,
   const Integer warmup,
   const Integer iterations )
{
   for ( Integer i = 0; i < warmup; ++i )
   {
      op( input, output );
   }

   const auto start = std::chrono::steady_clock::now();
   for ( Integer i = 0; i < iterations; ++i )
   {
      op( input, output );
   }
   const auto finish = std::chrono::steady_clock::now();
   return std::chrono::duration< double >( finish - start ).count() /
      static_cast< double >( iterations );
}

} // namespace

template < Integer order >
int RunMass(
   const Integer cells,
   const Integer warmup,
   const Integer iterations )
{
   Cartesian3DMesh mesh( 1.0, cells, cells, cells );
   auto finite_element = MakeLegendreFiniteElement(
      FiniteElementOrders< order, order, order >{} );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto integration_rule = MakeIntegrationRule(
      IntegrationRuleNumPoints< order + 2, order + 2, order + 2 >{} );
   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 3 > & ) -> Real
   {
      return 1.0;
   };

   auto reference_operator = MakeMassFiniteElementOperator<
      SerialKernelConfiguration >( fe_space, integration_rule, sigma );
   auto heterogeneous_operator = MakeMassFiniteElementOperator<
      K3HeterogeneousOpenMPConfiguration >(
         fe_space,
         integration_rule,
         sigma );

   const auto dofs = fe_space.GetNumberOfFiniteElementDofs();
   constexpr Integer reference_threads = 8;
   Vector input( dofs );
   Vector reference( dofs );
   Vector heterogeneous( dofs );
   input = 1.0;
   reference = 0.0;
   heterogeneous = 0.0;

   // Keep the reference team at the X100 worker count. The reference path is
   // an unbound host baseline; the heterogeneous path owns its placement.
   omp_set_num_threads( reference_threads );
   const double reference_seconds = TimeOperator(
      reference_operator, input, reference, warmup, iterations );
   const double heterogeneous_seconds = TimeOperator(
      heterogeneous_operator, input, heterogeneous, warmup, iterations );

   const Real * expected = reference.ReadHostData();
   const Real * actual = heterogeneous.ReadHostData();
   Real max_error = 0.0;
   for ( Integer i = 0; i < dofs; ++i )
   {
      max_error = std::max( max_error, std::abs( expected[ i ] - actual[ i ] ) );
   }

   std::cout << "K3 mass order: " << order << '\n';
   std::cout << "K3 mass cells per dimension: " << cells << '\n';
   std::cout << "K3 mass DoFs: " << dofs << '\n';
   std::cout << "K3 heterogeneous mass max error: " << max_error << '\n';
   const double reference_dofs_per_second =
      static_cast< double >( dofs ) / reference_seconds;
   const double heterogeneous_dofs_per_second =
      static_cast< double >( dofs ) / heterogeneous_seconds;
   std::cout << "K3 mass warmup iterations: " << warmup << '\n';
   std::cout << "K3 mass timed iterations: " << iterations << '\n';
   std::cout << "K3 mass reference threads: " << reference_threads << '\n';
   std::cout << "K3 mass reference seconds: " << reference_seconds << '\n';
   std::cout << "K3 mass heterogeneous seconds: "
             << heterogeneous_seconds << '\n';
   std::cout << "K3 mass reference DoF/s: "
             << reference_dofs_per_second << '\n';
   std::cout << "K3 mass heterogeneous DoF/s: "
             << heterogeneous_dofs_per_second << '\n';
   std::cout << "K3 mass heterogeneous speedup: "
             << heterogeneous_dofs_per_second / reference_dofs_per_second
             << '\n';
   if ( max_error > 1.0e-12 )
   {
      std::cerr << "K3 heterogeneous mass validation failed\n";
      return 1;
   }
   return 0;
}

int main()
{
   const Integer order = ReadPositiveEnvironment(
      "GENDIL_K3_MASS_ORDER", 1 );
   const Integer cells = ReadPositiveEnvironment(
      "GENDIL_K3_MASS_CELLS", 4 );
   const Integer warmup = ReadPositiveEnvironment(
      "GENDIL_K3_MASS_WARMUP", 1 );
   const Integer iterations = ReadPositiveEnvironment(
      "GENDIL_K3_MASS_ITERATIONS", 5 );

   switch ( order )
   {
      case 1:
         return RunMass< 1 >( cells, warmup, iterations );
      case 2:
         return RunMass< 2 >( cells, warmup, iterations );
      case 3:
         return RunMass< 3 >( cells, warmup, iterations );
      default:
         std::cerr << "GENDIL_K3_MASS_ORDER must be 1, 2, or 3\n";
         return 2;
   }
}
