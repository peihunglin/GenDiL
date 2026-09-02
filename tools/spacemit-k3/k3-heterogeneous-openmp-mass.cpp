// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace gendil;

int main()
{
   constexpr Integer order = 1;
   Cartesian3DMesh mesh( 1.0, 4, 4, 4 );
   auto finite_element = MakeLegendreFiniteElement(
      FiniteElementOrders< order, order, order >{} );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto integration_rule = MakeIntegrationRule(
      IntegrationRuleNumPoints< 3, 3, 3 >{} );
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
   Vector input( dofs );
   Vector reference( dofs );
   Vector heterogeneous( dofs );
   input = 1.0;
   reference = 0.0;
   heterogeneous = 0.0;

   reference_operator( input, reference );
   heterogeneous_operator( input, heterogeneous );

   const Real * expected = reference.ReadHostData();
   const Real * actual = heterogeneous.ReadHostData();
   Real max_error = 0.0;
   for ( Integer i = 0; i < dofs; ++i )
   {
      max_error = std::max( max_error, std::abs( expected[ i ] - actual[ i ] ) );
   }

   std::cout << "K3 heterogeneous mass max error: " << max_error << '\n';
   if ( max_error > 1.0e-12 )
   {
      std::cerr << "K3 heterogeneous mass validation failed\n";
      return 1;
   }
   return 0;
}
