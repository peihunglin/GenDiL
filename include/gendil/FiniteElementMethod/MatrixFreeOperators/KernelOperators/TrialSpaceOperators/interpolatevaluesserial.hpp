// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/getrank.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevaluesthreaded.hpp"

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
#include <cstdint>
#include <type_traits>
#endif

namespace gendil
{

namespace details
{

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
namespace k3
{
   // Phase-1 precision policy: uniform FP16 storage for both X100 and A100
   using Storage = __fp16;

   inline Storage ToStorage(Real v)
   {
      return static_cast<Storage>(v);
   }
   inline Real FromStorage(Storage v)
   {
      return static_cast<Real>(v);
   }

   // Simple A100 check via thread-local flag in K3HeterogeneousOpenMPConfiguration
   inline bool IsA100()
   {
      // Forward declaration, real implementation will query K3 config
      return false;
   }

   template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
   GENDIL_HOST_DEVICE
   auto InterpContractionIME( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
   {
      // Skeleton: tile-friendly path using FP16 storage
      // Real implementation will pack B.values/q,d and u into FP16 tiles
      // and issue smt.vfwmadot for FP16xFP16->FP32 accumulation.
      // For now, fall back to scalar to preserve correctness.
      constexpr Integer ND = domain_dim_v< Op1D >;
      SerialRecursiveArray< Real, contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu{};

      Loop< contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >(
         [&] ( auto ... indices_ )
         {
            auto indices = std::make_tuple( indices_ ... );
            const Integer q = std::get< ActiveDim >( indices );
            Real value = 0.0;
            auto& d = std::get< ActiveDim >( indices );
            for ( d = 0; d < ND; ++d )
            {
               const Real dof = u( std::get< Is >( indices ) ... );
               if constexpr ( Gradient )
               {
                  const Real g = B.gradients( q, d );
                  value += g * dof;
               }
               else
               {
                  const Real b = B.values( q, d );
                  value += b * dof;
               }
            }
            Bu( indices_ ... ) = value;
         }
      );
      return Bu;
   }
} // namespace k3
#endif

template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
GENDIL_HOST_DEVICE
auto InterpContraction( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
{
   SerialRecursiveArray< Real, contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu{};

   constexpr Integer ND = domain_dim_v< Op1D >;

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
   // Phase-1 gate: ND >=8 and A100 present
   if constexpr ( ND >= 8 )
   {
      if ( k3::IsA100() )
      {
         auto res = k3::InterpContractionIME< Gradient, ActiveDim >( u, B, std::index_sequence< Is ... >{} );
         return res;
      }
   }
#endif

   Loop< contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >(
      [&] ( auto ... indices_ )
      {
         // TODO: make_tuple copies all of the indices however all but one index
         // (ActiveDim) needs to be modifiable. The other indices could be tied
         // instead. If the compiler does not recognize this, then this
         // implementation may use twice as many registers as needed.
         auto indices = std::make_tuple( indices_ ... );
         const Integer q = std::get< ActiveDim >( indices );

         Real value = 0.0;
         auto& d = std::get< ActiveDim >( indices );
         for ( d = 0; d < ND; ++d )
         {
            const Real dof = u( std::get< Is >( indices ) ... );

            // TODO: Replace with std::function< Real( LocalIndex, LocalIndex ) > ?
            if constexpr ( Gradient )
            {
               const Real g = B.gradients( q, d );
               value += g * dof;
            }
            else
            {
               const Real b = B.values( q, d );
               value += b * dof;
            }
         }

         Bu( indices_ ... ) = value;
      }
   );

   return Bu;
}

template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D >
GENDIL_HOST_DEVICE
inline auto InterpContraction( InputTensor const & u, Op1D const & B )
{
   constexpr Integer Rank = get_rank_v< InputTensor >;
   return InterpContraction< Gradient, ActiveDim >( u, B, std::make_index_sequence< Rank >{} );
}

template < Integer ActiveDim,
           typename ElementDofToQuad,
           typename InputTensor>
GENDIL_HOST_DEVICE
inline auto InterpolateValuesImpl( const ElementDofToQuad & element_quad_data, const InputTensor & u )
{
   constexpr Integer Rank = get_rank_v< InputTensor >;
   static_assert( ActiveDim < Rank );

   auto& B = GetTensorProductEntry<ActiveDim>(element_quad_data);

   if constexpr ( ActiveDim+1 == Rank )
      return InterpContraction< false, ActiveDim >( u, B );
   else
      return InterpContraction< false, ActiveDim >( InterpolateValuesImpl< ActiveDim+1 >( element_quad_data, u ), B );
}

} // namespace details

template <
   typename ElementDofToQuad,
   typename DofTensor >
GENDIL_HOST_DEVICE
auto InterpolateValuesSerial(
   const ElementDofToQuad & element_quad_data,
   const DofTensor & u )
{
   return details::InterpolateValuesImpl< 0 >( element_quad_data, u );
}

} // namespace gendil
