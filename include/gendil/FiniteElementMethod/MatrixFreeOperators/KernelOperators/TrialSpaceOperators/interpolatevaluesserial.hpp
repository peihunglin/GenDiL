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
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/k3ime.hpp"
#endif

#if defined(GENDIL_ENABLE_K3_FP16_BASELINE)
#include <cstdint>
#include <type_traits>
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
#endif

namespace gendil
{

namespace details
{

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
namespace k3
{
    template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
   GENDIL_HOST_DEVICE
   auto InterpContractionIME( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
   {
       constexpr Integer ND = domain_dim_v< Op1D >;
       constexpr Integer NQ = range_dim_v< Op1D >;
       constexpr Integer Rank = sizeof...( Is );
       constexpr std::array< Integer, Rank > input_extents = {
          get_tensor_size_v< Is, InputTensor > ... };
       Integer batch_size = 1;
       for ( Integer dim = 0; dim < Rank; ++dim )
       {
          if ( dim != ActiveDim )
             batch_size *= input_extents[ dim ];
       }
       SerialRecursiveArray< Real, contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu{};
       for ( Integer q0 = 0; q0 < NQ; q0 += TILE_M )
       {
          for ( Integer n0 = 0; n0 < batch_size; n0 += TILE_N )
          {
             float accumulator[ TILE_M * TILE_N ]{};
             for ( Integer d0 = 0; d0 < ND; d0 += TILE_K )
             {
                Storage lhs[ TILE_M * TILE_K ]{};
                Storage rhs_transposed[ TILE_N * TILE_K ]{};
                for ( Integer m = 0; m < TILE_M && q0 + m < NQ; ++m )
                {
                   for ( Integer k = 0; k < TILE_K && d0 + k < ND; ++k )
                   {
                      if constexpr ( Gradient )
                         lhs[ m * TILE_K + k ] = ToStorage( B.gradients( q0 + m, d0 + k ) );
                      else
                         lhs[ m * TILE_K + k ] = ToStorage( B.values( q0 + m, d0 + k ) );
                   }
                }
                for ( Integer n = 0; n < TILE_N && n0 + n < batch_size; ++n )
                {
                   std::array< Integer, Rank > indices{};
                   Integer flat = n0 + n;
                   for ( Integer dim = Rank; dim-- > 0; )
                   {
                      if ( dim != ActiveDim )
                      {
                         indices[ dim ] = flat % input_extents[ dim ];
                         flat /= input_extents[ dim ];
                      }
                   }
                   for ( Integer k = 0; k < TILE_K && d0 + k < ND; ++k )
                   {
                      indices[ ActiveDim ] = d0 + k;
                      rhs_transposed[ n * TILE_K + k ] = ToStorage( std::apply( u, indices ) );
                   }
                }
                MultiplyAccumulate8x8x8( lhs, rhs_transposed, accumulator );
             }
             for ( Integer m = 0; m < TILE_M && q0 + m < NQ; ++m )
             {
                for ( Integer n = 0; n < TILE_N && n0 + n < batch_size; ++n )
                {
                   std::array< Integer, Rank > indices{};
                   Integer flat = n0 + n;
                   for ( Integer dim = Rank; dim-- > 0; )
                   {
                      if ( dim != ActiveDim )
                      {
                         indices[ dim ] = flat % input_extents[ dim ];
                         flat /= input_extents[ dim ];
                      }
                   }
                   indices[ ActiveDim ] = q0 + m;
                   std::apply( [&] ( auto ... output_indices )
                   {
                      Bu( output_indices ... ) = static_cast< Real >( accumulator[ m * TILE_N + n ] );
                   }, indices );
                }
             }
          }
       }
       return Bu;
   }

} // namespace k3
#endif

#if defined(GENDIL_ENABLE_K3_FP16_BASELINE)
namespace k3_fp16_baseline
{
   // FP16 baseline policy for A100 scalar path
   using Storage = _Float16;

   inline Storage ToStorage(Real v)
   {
      return static_cast<Storage>(v);
   }
   inline Real FromStorage(Storage v)
   {
      return static_cast<Real>(v);
   }

   template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
   GENDIL_HOST_DEVICE
   auto InterpContractionScalarFP16( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
   {
      constexpr Integer ND = domain_dim_v< Op1D >;
      SerialRecursiveArray< Real, contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu{};

      Loop< contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >(
         [&] ( auto ... indices_ )
         {
            auto indices = std::make_tuple( indices_ ... );
            const Integer q = std::get< ActiveDim >( indices );
            using StorageT = _Float16;
            StorageT value = StorageT(0.0);
            auto& d = std::get< ActiveDim >( indices );
            for ( Integer dd = 0; dd < ND; ++dd )
            {
               const Real dof_real = u( std::get< Is >( indices ) ... );
               const StorageT dof = static_cast<StorageT>(dof_real);
               if constexpr ( Gradient )
               {
                  const Real g = B.gradients( q, dd );
                  const StorageT g_s = static_cast<StorageT>(g);
                  value += static_cast<StorageT>( static_cast<float>(g_s) * static_cast<float>(dof) );
               }
               else
               {
                  const Real b = B.values( q, dd );
                  const StorageT b_s = static_cast<StorageT>(b);
                  value += static_cast<StorageT>( static_cast<float>(b_s) * static_cast<float>(dof) );
               }
            }
            Bu( indices_ ... ) = static_cast<Real>(value);
         }
      );
      return Bu;
   }
}
#endif

template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
GENDIL_HOST_DEVICE
auto InterpContraction( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
{
   SerialRecursiveArray< Real, contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu{};

   constexpr Integer ND = domain_dim_v< Op1D >;

#if defined(GENDIL_ENABLE_K3_FP16_BASELINE)
   // FP16 baseline gate: ND >=8 and A100 present
   if constexpr ( ND >= 8 )
   {
      if ( K3HeterogeneousOpenMPConfiguration::OnA100() )
      {
         auto res = k3_fp16_baseline::InterpContractionScalarFP16< Gradient, ActiveDim, InputTensor, Op1D, Is... >( u, B, std::index_sequence< Is ... >{} );
         return res;
      }
   }
#endif

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
   // Phase-1 gate: ND >=8 and A100 present
   if constexpr ( ND >= 8 )
   {
      if ( k3::IsA100() )
      {
         auto res = k3::InterpContractionIME< Gradient, ActiveDim, InputTensor, Op1D, Is... >( u, B, std::index_sequence< Is ... >{} );
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
