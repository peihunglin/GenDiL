// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file realtype.hpp
 * @brief Precision-aware helpers for gendil::Real.
 *
 * C++20 does not specify standard-library support for compiler extended types
 * such as `_Float16` and `__bf16`. GenDiL deliberately does not add overloads
 * or specializations to `namespace std` for those compiler types. Use the
 * helpers below, plus `gendil::Sqrt` and `gendil::Abs`, in generic GenDiL code.
 */

#include <cmath>
#include <cstdint>
#include <limits>

#include "gendil/Utilities/types.hpp"

namespace gendil {

namespace internal {

#if GENDIL_LOW_PRECISION_REAL
/// Compile-time description of the low precision real types.
template < typename T >
struct LowPrecisionTraits;

/// IEEE 754 binary16: 1 sign bit, 5 exponent bits, 10 fraction bits.
template < >
struct LowPrecisionTraits< _Float16 >
{
   using type = _Float16;
   static constexpr const char * name = "fp16";
   static constexpr int digits = 11;
   static constexpr int digits10 = 3;
   static constexpr int max_digits10 = 5;
   static constexpr int min_exponent = -13;
   static constexpr int min_exponent10 = -4;
   static constexpr int max_exponent = 16;
   static constexpr int max_exponent10 = 4;
   static constexpr type min() noexcept { return static_cast< type >( 0x1p-14f ); }
   static constexpr type max() noexcept { return static_cast< type >( 65504.0f ); }
   static constexpr type epsilon() noexcept { return static_cast< type >( 0x1p-10f ); }
   static constexpr type denorm_min() noexcept { return static_cast< type >( 0x1p-24f ); }
};

/// bfloat16: 1 sign bit, 8 exponent bits, 7 fraction bits.
template < >
struct LowPrecisionTraits< __bf16 >
{
   using type = __bf16;
   static constexpr const char * name = "bf16";
   static constexpr int digits = 8;
   static constexpr int digits10 = 2;
   static constexpr int max_digits10 = 4;
   static constexpr int min_exponent = -125;
   static constexpr int min_exponent10 = -37;
   static constexpr int max_exponent = 128;
   static constexpr int max_exponent10 = 38;
   static constexpr type min() noexcept { return static_cast< type >( 0x1p-126f ); }
   static constexpr type max() noexcept { return static_cast< type >( 0x1.FEp127f ); }
   static constexpr type epsilon() noexcept { return static_cast< type >( 0x1p-7f ); }
   static constexpr type denorm_min() noexcept { return static_cast< type >( 0x1p-133f ); }
};

using RealTraits = LowPrecisionTraits< Real >;

#endif

} // namespace internal

} // namespace gendil

namespace gendil {

/// True when gendil::Real is one of the low precision types (FP16 or BF16).
inline constexpr bool is_low_precision_real_v = GENDIL_LOW_PRECISION_REAL;

/// Number of significand bits of gendil::Real (including the implicit bit).
#if GENDIL_LOW_PRECISION_REAL
inline constexpr int real_significand_digits_v = internal::RealTraits::digits;
constexpr Real RealMinNormal() { return internal::RealTraits::min(); }
constexpr Real RealMax() { return internal::RealTraits::max(); }
constexpr Real RealEpsilonValue() { return internal::RealTraits::epsilon(); }
#else
inline constexpr int real_significand_digits_v = std::numeric_limits< Real >::digits;
constexpr Real RealMinNormal() { return std::numeric_limits< Real >::min(); }
constexpr Real RealMax() { return std::numeric_limits< Real >::max(); }
constexpr Real RealEpsilonValue() { return std::numeric_limits< Real >::epsilon(); }
#endif

/// Human readable name of the active real type.
constexpr const char * RealTypeName()
{
#if defined( GENDIL_REAL_TYPE_FP16 )
   return "fp16";
#elif defined( GENDIL_REAL_TYPE_BF16 )
   return "bf16";
#elif defined( GENDIL_REAL_TYPE_FLOAT )
   return "float";
#else
   return "double";
#endif
}

/// Machine epsilon of gendil::Real as a double.
constexpr double RealEpsilon()
{
   return static_cast< double >( RealEpsilonValue() );
}

namespace internal {

/// Natural logarithm usable in constant expressions (x > 0).
constexpr double ConstexprLog( double x )
{
   constexpr double ln2 = 0.6931471805599453;
   int exponent = 0;
   while ( x >= 1.5 ) { x *= 0.5; ++exponent; }
   while ( x < 0.75 ) { x *= 2.0; --exponent; }
   // ln(x) = 2 atanh( (x-1)/(x+1) ), |u| <= 0.2 after the reduction.
   const double u = ( x - 1.0 ) / ( x + 1.0 );
   const double u2 = u * u;
   double term = u;
   double sum = 0.0;
   for ( int k = 1; k < 64; k += 2 )
   {
      sum += term / k;
      term *= u2;
   }
   return 2.0 * sum + exponent * ln2;
}

/// Exponential usable in constant expressions.
constexpr double ConstexprExp( double y )
{
   constexpr double ln2 = 0.6931471805599453;
   const int k = static_cast< int >( y / ln2 + ( y >= 0.0 ? 0.5 : -0.5 ) );
   const double r = y - k * ln2;
   double term = 1.0;
   double sum = 1.0;
   for ( int n = 1; n < 32; ++n )
   {
      term *= r / n;
      sum += term;
   }
   double scale = 1.0;
   for ( int i = 0; i < ( k < 0 ? -k : k ); ++i )
   {
      scale *= ( k < 0 ) ? 0.5 : 2.0;
   }
   return sum * scale;
}

} // namespace internal

/**
 * @brief Rescales a tolerance calibrated for double precision to the active
 *        real type.
 *
 * Tests and examples express tolerances as absolute numbers that assume
 * `Real == double` (e.g. `1e-12`). Such a tolerance demands a fixed fraction
 * of the digits that double precision offers; this function demands the same
 * fraction of the digits of the active type:
 *
 *     RealTolerance( t ) = t ^ ( digits( Real ) / digits( double ) )
 *
 * where `digits` is the significand width in bits. In double precision the
 * tolerance is returned unchanged. Representative values:
 *
 *     t       float     fp16     bf16
 *     1e-8    2.4e-4    2.2e-2   6.0e-2
 *     1e-10   8.9e-5    8.5e-3   3.0e-2
 *     1e-12   3.6e-6    3.0e-3   1.5e-2
 *     1e-14   1.1e-6    1.3e-3   7.7e-3
 *
 * The result is a double rather than a Real so that it can be combined with
 * either type without overflowing the narrow range of FP16. Non-positive
 * inputs are returned unchanged.
 *
 * @param double_precision_tolerance Tolerance written for double precision.
 * @return The equivalent tolerance for gendil::Real.
 */
constexpr double RealTolerance( double double_precision_tolerance )
{
   constexpr double exponent =
       static_cast< double >( real_significand_digits_v ) /
      static_cast< double >( std::numeric_limits< double >::digits );
   if constexpr ( exponent == 1.0 )
   {
      return double_precision_tolerance;
   }
   else
   {
      if ( !( double_precision_tolerance > 0.0 ) )
      {
         return double_precision_tolerance;
      }
      return internal::ConstexprExp(
         exponent * internal::ConstexprLog( double_precision_tolerance ) );
   }
}

} // namespace gendil
