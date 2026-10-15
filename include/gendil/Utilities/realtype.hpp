// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file realtype.hpp
 * @brief Precision-aware helpers for gendil::Real and standard library support
 *        for the low precision (FP16 / BF16) real types.
 *
 * In C++20 the standard library knows nothing about `_Float16` and `__bf16`:
 * `std::numeric_limits` is unspecialized (every member is zero), the `<cmath>`
 * functions are ambiguous (the value converts equally well to `float`,
 * `double`, and `long double`), and streams cannot print the values. This
 * header fills those gaps when GenDiL is configured with
 * `GENDIL_REAL_TYPE=FP16` or `GENDIL_REAL_TYPE=BF16` so that generic code
 * written against `Real` and the standard library keeps compiling.
 *
 * The additions to namespace `std` are limited to the single low precision
 * type in use, mirror what C++23 `<stdfloat>` provides for `std::float16_t`
 * and `std::bfloat16_t`, and are disabled when the standard library already
 * ships that support (`__STDCPP_FLOAT16_T__` / `__STDCPP_BFLOAT16_T__`).
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>

#include "gendil/Utilities/types.hpp"

#if GENDIL_LOW_PRECISION_REAL

#if defined( GENDIL_REAL_TYPE_FP16 ) && defined( __STDCPP_FLOAT16_T__ )
#define GENDIL_STD_HAS_LOW_PRECISION_REAL 1
#elif defined( GENDIL_REAL_TYPE_BF16 ) && defined( __STDCPP_BFLOAT16_T__ )
#define GENDIL_STD_HAS_LOW_PRECISION_REAL 1
#else
#define GENDIL_STD_HAS_LOW_PRECISION_REAL 0
#endif

namespace gendil {

namespace internal {

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

} // namespace internal

} // namespace gendil

#if !GENDIL_STD_HAS_LOW_PRECISION_REAL

namespace std {

template < >
class numeric_limits< gendil::Real >
{
   using traits = gendil::internal::RealTraits;

public:
   static constexpr bool is_specialized = true;
   static constexpr bool is_signed = true;
   static constexpr bool is_integer = false;
   static constexpr bool is_exact = false;
   static constexpr bool has_infinity = true;
   static constexpr bool has_quiet_NaN = true;
   static constexpr bool has_signaling_NaN = true;
   static constexpr float_denorm_style has_denorm = denorm_present;
   static constexpr bool has_denorm_loss = false;
   static constexpr float_round_style round_style = round_to_nearest;
   static constexpr bool is_iec559 = true;
   static constexpr bool is_bounded = true;
   static constexpr bool is_modulo = false;
   static constexpr int digits = traits::digits;
   static constexpr int digits10 = traits::digits10;
   static constexpr int max_digits10 = traits::max_digits10;
   static constexpr int radix = 2;
   static constexpr int min_exponent = traits::min_exponent;
   static constexpr int min_exponent10 = traits::min_exponent10;
   static constexpr int max_exponent = traits::max_exponent;
   static constexpr int max_exponent10 = traits::max_exponent10;
   static constexpr bool traps = false;
   static constexpr bool tinyness_before = false;

   static constexpr gendil::Real min() noexcept { return traits::min(); }
   static constexpr gendil::Real lowest() noexcept { return -traits::max(); }
   static constexpr gendil::Real max() noexcept { return traits::max(); }
   static constexpr gendil::Real epsilon() noexcept { return traits::epsilon(); }
   static constexpr gendil::Real round_error() noexcept { return static_cast< gendil::Real >( 0.5f ); }
   static constexpr gendil::Real infinity() noexcept { return static_cast< gendil::Real >( __builtin_inff() ); }
   static constexpr gendil::Real quiet_NaN() noexcept { return static_cast< gendil::Real >( __builtin_nanf( "" ) ); }
   static constexpr gendil::Real signaling_NaN() noexcept { return static_cast< gendil::Real >( __builtin_nansf( "" ) ); }
   static constexpr gendil::Real denorm_min() noexcept { return traits::denorm_min(); }
};

template < >
class numeric_limits< const gendil::Real > : public numeric_limits< gendil::Real > { };
template < >
class numeric_limits< volatile gendil::Real > : public numeric_limits< gendil::Real > { };
template < >
class numeric_limits< const volatile gendil::Real > : public numeric_limits< gendil::Real > { };

// Unary <cmath> functions evaluated in single precision and rounded back.
#define GENDIL_LOW_PRECISION_UNARY_MATH( name ) \
   inline gendil::Real name( gendil::Real x ) \
   { \
      return static_cast< gendil::Real >( name( static_cast< float >( x ) ) ); \
   }

GENDIL_LOW_PRECISION_UNARY_MATH( abs )
GENDIL_LOW_PRECISION_UNARY_MATH( fabs )
GENDIL_LOW_PRECISION_UNARY_MATH( sqrt )
GENDIL_LOW_PRECISION_UNARY_MATH( cbrt )
GENDIL_LOW_PRECISION_UNARY_MATH( exp )
GENDIL_LOW_PRECISION_UNARY_MATH( exp2 )
GENDIL_LOW_PRECISION_UNARY_MATH( expm1 )
GENDIL_LOW_PRECISION_UNARY_MATH( log )
GENDIL_LOW_PRECISION_UNARY_MATH( log2 )
GENDIL_LOW_PRECISION_UNARY_MATH( log10 )
GENDIL_LOW_PRECISION_UNARY_MATH( log1p )
GENDIL_LOW_PRECISION_UNARY_MATH( sin )
GENDIL_LOW_PRECISION_UNARY_MATH( cos )
GENDIL_LOW_PRECISION_UNARY_MATH( tan )
GENDIL_LOW_PRECISION_UNARY_MATH( asin )
GENDIL_LOW_PRECISION_UNARY_MATH( acos )
GENDIL_LOW_PRECISION_UNARY_MATH( atan )
GENDIL_LOW_PRECISION_UNARY_MATH( sinh )
GENDIL_LOW_PRECISION_UNARY_MATH( cosh )
GENDIL_LOW_PRECISION_UNARY_MATH( tanh )
GENDIL_LOW_PRECISION_UNARY_MATH( floor )
GENDIL_LOW_PRECISION_UNARY_MATH( ceil )
GENDIL_LOW_PRECISION_UNARY_MATH( round )
GENDIL_LOW_PRECISION_UNARY_MATH( trunc )

#undef GENDIL_LOW_PRECISION_UNARY_MATH

// Binary <cmath> functions. Mixed overloads with the standard floating-point
// types are provided so that expressions such as `std::pow( x, 2.0 )` stay
// unambiguous; they return the wider type, as the usual arithmetic
// conversions would.
#define GENDIL_LOW_PRECISION_BINARY_MATH( name ) \
   inline gendil::Real name( gendil::Real x, gendil::Real y ) \
   { \
      return static_cast< gendil::Real >( \
         name( static_cast< float >( x ), static_cast< float >( y ) ) ); \
   } \
   inline float name( gendil::Real x, float y ) \
   { \
      return name( static_cast< float >( x ), y ); \
   } \
   inline float name( float x, gendil::Real y ) \
   { \
      return name( x, static_cast< float >( y ) ); \
   } \
   inline double name( gendil::Real x, double y ) \
   { \
      return name( static_cast< double >( x ), y ); \
   } \
   inline double name( double x, gendil::Real y ) \
   { \
      return name( x, static_cast< double >( y ) ); \
   } \
   inline gendil::Real name( gendil::Real x, int y ) \
   { \
      return static_cast< gendil::Real >( \
         name( static_cast< float >( x ), static_cast< float >( y ) ) ); \
   }

GENDIL_LOW_PRECISION_BINARY_MATH( pow )
GENDIL_LOW_PRECISION_BINARY_MATH( atan2 )
GENDIL_LOW_PRECISION_BINARY_MATH( fmod )
GENDIL_LOW_PRECISION_BINARY_MATH( hypot )
GENDIL_LOW_PRECISION_BINARY_MATH( fmax )
GENDIL_LOW_PRECISION_BINARY_MATH( fmin )
GENDIL_LOW_PRECISION_BINARY_MATH( fdim )

#undef GENDIL_LOW_PRECISION_BINARY_MATH

inline gendil::Real fma( gendil::Real x, gendil::Real y, gendil::Real z )
{
   return static_cast< gendil::Real >(
      fma( static_cast< float >( x ), static_cast< float >( y ), static_cast< float >( z ) ) );
}

inline bool isnan( gendil::Real x ) { return isnan( static_cast< float >( x ) ); }
inline bool isinf( gendil::Real x ) { return isinf( static_cast< float >( x ) ); }
inline bool isfinite( gendil::Real x ) { return isfinite( static_cast< float >( x ) ); }
inline bool isnormal( gendil::Real x ) { return isnormal( static_cast< float >( x ) ); }
inline bool signbit( gendil::Real x ) { return signbit( static_cast< float >( x ) ); }

inline string to_string( gendil::Real x ) { return to_string( static_cast< float >( x ) ); }

/// Stream insertion prints the value through its exact single precision image.
template < typename CharT, typename Traits >
basic_ostream< CharT, Traits > & operator<<( basic_ostream< CharT, Traits > & os, gendil::Real x )
{
   return os << static_cast< float >( x );
}

/// Stream extraction reads a float and rounds it to the low precision type.
template < typename CharT, typename Traits >
basic_istream< CharT, Traits > & operator>>( basic_istream< CharT, Traits > & is, gendil::Real & x )
{
   float value;
   is >> value;
   x = static_cast< gendil::Real >( value );
   return is;
}

} // namespace std

#endif // !GENDIL_STD_HAS_LOW_PRECISION_REAL

#endif // GENDIL_LOW_PRECISION_REAL

namespace gendil {

/// True when gendil::Real is one of the low precision types (FP16 or BF16).
inline constexpr bool is_low_precision_real_v = GENDIL_LOW_PRECISION_REAL;

/// Number of significand bits of gendil::Real (including the implicit bit).
inline constexpr int real_significand_digits_v = std::numeric_limits< Real >::digits;

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
   return static_cast< double >( std::numeric_limits< Real >::epsilon() );
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
      static_cast< double >( std::numeric_limits< Real >::digits ) /
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
