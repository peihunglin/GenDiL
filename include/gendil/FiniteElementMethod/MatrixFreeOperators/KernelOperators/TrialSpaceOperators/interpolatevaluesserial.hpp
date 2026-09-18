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
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
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

   inline bool IsA100()
   {
      return gendil::KernelContext::K3HeterogeneousOpenMPConfiguration::OnA100();
   }

   // Tile parameters for Xsmtfp16fp32mm 8x8x8
   constexpr int TILE_M = 8;
   constexpr int TILE_N = 8;
   constexpr int TILE_K = 8;

   template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
   GENDIL_HOST_DEVICE
   inline void InterpContractionIMEBlock(
      InputTensor const & u,
      Op1D const & B,
      const std::array<Integer, 4>& idx,
      const Integer q_start,
      const Integer k_start,
      Real* out )
   {
      // Block size 8x8: q in [q_start, q_start+7], k in [k_start, k_start+7]
      // Accumulate into out[8][8] as FP32
      // This is a scalar fallback that mimics tile accumulation.
      // Real IME will use smt.vfwmadot with FP16 packed buffers.
      for (int i = 0; i < TILE_M; ++i)
      {
         const Integer q = q_start + i;
         Real acc = 0.0;
         // For demo, we accumulate a single k element
         const Integer d = k_start;
         const Real dof = u(idx[0], idx[1], idx[2], idx[3]); // placeholder
         if constexpr (Gradient)
         {
            const Real g = B.gradients(q, d);
            acc += g * dof;
         }
         else
         {
            const Real b = B.values(q, d);
            acc += b * dof;
         }
         out[i] = acc;
      }
    }

    template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
   GENDIL_HOST_DEVICE
   auto InterpContractionIME( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
   {
      // Tile-friendly path using FP16 storage
      // For pilot, we keep scalar correctness but layout is tile aware.
      // Real IME will pack B.values/q,d and u into FP16 tiles
      // and issue smt.vfwmadot for FP16xFP16->FP32 accumulation.
      constexpr Integer ND = domain_dim_v< Op1D >;
      SerialRecursiveArray< Real, contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu{};

      Loop< contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >(
         [&] ( auto ... indices_ )
         {
            auto indices = std::make_tuple( indices_ ... );
            const Integer q = std::get< ActiveDim >( indices );
            Real value = 0.0;
            auto& d = std::get< ActiveDim >( indices );
            // Tile loops over d in blocks of TILE_K
            for ( Integer d0 = 0; d0 < ND; d0 += TILE_K )
            {
               const Integer d_end = std::min<Integer>( d0 + TILE_K, ND );
               for ( Integer dd = d0; dd < d_end; ++dd )
               {
                  const Real dof = u( std::get< Is >( indices ) ... );
                  if constexpr ( Gradient )
                  {
                     const Real g = B.gradients( q, dd );
                     value += g * dof;
                  }
                  else
                  {
                     const Real b = B.values( q, dd );
                     value += b * dof;
                  }
               }
            }
            Bu( indices_ ... ) = value;
         }
      );
      return Bu;
   }

} // namespace k3
#endif

#if defined(GENDIL_ENABLE_K3_FP16_BASELINE)
namespace k3_fp16_baseline
{
   // FP16 baseline policy for A100 scalar path
   using Storage = __fp16;

   inline Storage ToStorage(Real v)
   {
      return static_cast<Storage>(v);
   }
   inline Real FromStorage(Storage v)
   {
      return static_cast<Real>(v);
   }

   inline bool IsA100()
   {
      return gendil::KernelContext::K3HeterogeneousOpenMPConfiguration::OnA100();
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
            using StorageT = __fp16;
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
      if ( k3_fp16_baseline::IsA100() )
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
