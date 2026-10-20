// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Verifies the GenDiL-owned precision helpers used when gendil::Real is one
// of the compiler extended 16-bit floating-point types.

#include <gendil/gendil.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

using namespace gendil;

namespace
{

int failures = 0;

void Check( const bool condition, const std::string & what )
{
   if ( !condition )
   {
      ++failures;
      std::cout << "FAILED: " << what << "\n";
   }
}

bool TestNumericLimits()
{
   Check( real_significand_digits_v > 0, "significand digits exposed" );

   // 1 + epsilon is the smallest representable number above one.
   const Real one = Real( 1 );
   const Real eps = RealEpsilonValue();
   const Real half_eps = static_cast< Real >( static_cast< double >( eps ) * 0.5 );
   Check( one + eps > one, "1 + epsilon > 1" );
   Check( one + half_eps == one, "1 + epsilon / 2 == 1" );

   // The reported extrema are finite and bracket the representable range.
   Check( std::isfinite( static_cast< long double >( RealMax() ) ), "max is finite" );
   Check( RealMinNormal() > Real( 0 ), "min is positive" );
   Check( RealMinNormal() < Real( 1 ), "min is less than one" );

   if constexpr ( is_low_precision_real_v )
   {
      // Doubling the maximum overflows, halving denorm_min underflows.
       const Real two_max = static_cast< Real >( static_cast< float >( RealMax() ) * 2.0f );
       Check( std::isinf( two_max ), "2 * max overflows" );
       Check( sizeof( Real ) == 2, "low precision Real is 16 bits wide" );
   }
   return failures == 0;
}

bool TestMathFunctions()
{
   const Real x = Real( 2.25 );
   Check( Sqrt( x ) == Real( 1.5 ), "gendil::Sqrt" );
   Check( Abs( Real( -1.5 ) ) == Real( 1.5 ), "gendil::Abs" );
   Check( Pow< 3 >( Real( 2 ) ) == Real( 8 ), "gendil::Pow" );
   Check( std::max( x, Real( 1 ) ) == x, "std::max" );
   Check( std::min( x, Real( 1 ) ) == Real( 1 ), "std::min" );
   Check( Max( x, Real( 1 ) ) == x, "gendil::Max" );
   Check( Min( x, Real( 1 ) ) == Real( 1 ), "gendil::Min" );
   return failures == 0;
}

bool TestRealTolerance()
{
   // RealTolerance preserves the fraction of available digits that a double
   // precision tolerance demands, and is the identity in double precision.
   const double exponent =
      static_cast< double >( real_significand_digits_v ) /
      static_cast< double >( std::numeric_limits< double >::digits );
   for ( const double tol : { 1e-6, 1e-8, 1e-10, 1e-11, 1e-12, 1e-14, 2.5e-9 } )
   {
      const double expected = std::pow( tol, exponent );
      const double computed = RealTolerance( tol );
      Check( std::abs( computed - expected ) <= 1e-12 * expected,
             "RealTolerance(" + std::to_string( tol ) + ") matches std::pow" );
      if constexpr ( std::is_same_v< Real, double > )
      {
         Check( computed == tol, "RealTolerance is the identity in double precision" );
      }
      else
      {
         Check( computed > tol, "RealTolerance loosens the tolerance in reduced precision" );
         Check( computed < 1.0, "RealTolerance stays below one" );
      }
   }
   static_assert( RealTolerance( 1e-12 ) > 0.0, "RealTolerance is usable in constant expressions" );
   Check( RealTolerance( 0.0 ) == 0.0, "RealTolerance(0) == 0" );
   Check( RealTolerance( 1e-12 ) < RealTolerance( 1e-10 ), "RealTolerance is monotone" );
   Check( RealEpsilon() == static_cast< double >( RealEpsilonValue() ), "RealEpsilon" );
   Check( std::string( RealTypeName() ).size() > 0, "RealTypeName" );
   return failures == 0;
}

bool TestAtomicAdd()
{
   Real value = Real( 0 );
   const Real old = AtomicAdd( value, Real( 1.5 ) );
   Check( old == Real( 0 ), "AtomicAdd returns the previous value" );
   Check( value == Real( 1.5 ), "AtomicAdd updates the value" );
   AtomicAddInPlace( value, Real( 0.5 ) );
   Check( value == Real( 2 ), "AtomicAddInPlace updates the value" );

   // Adjacent elements must not disturb each other.
   Real pair[2] = { Real( 1 ), Real( 10 ) };
   AtomicAddInPlace( pair[0], Real( 1 ) );
   AtomicAddInPlace( pair[1], Real( -1 ) );
   Check( pair[0] == Real( 2 ) && pair[1] == Real( 9 ), "AtomicAddInPlace on adjacent elements" );
   return failures == 0;
}

bool TestVectorRandomFill()
{
   Vector v( 64 );
   FillRandom( v, 7, Real( -1 ), Real( 1 ) );
   bool in_range = true;
   bool nonzero = false;
   for ( size_t i = 0; i < v.Size(); ++i )
   {
      in_range = in_range && v[i] >= Real( -1 ) && v[i] <= Real( 1 );
      nonzero = nonzero || v[i] != Real( 0 );
   }
   Check( in_range && nonzero, "FillRandom produces values in range" );
   Check( std::isfinite( static_cast< float >( Norml2( v ) ) ), "Norml2 of a random vector is finite" );
   return failures == 0;
}

} // namespace

int main()
{
   std::cout << "Real type: " << RealTypeName()
             << " (significand digits: " << real_significand_digits_v
             << ", epsilon: " << RealEpsilon()
             << ", RealTolerance(1e-12): " << RealTolerance( 1e-12 ) << ")\n";

   TestNumericLimits();
   TestMathFunctions();
   TestRealTolerance();
   TestAtomicAdd();
   TestVectorRandomFill();

   if ( failures != 0 )
   {
      std::cout << failures << " check(s) failed.\n";
      return 1;
   }
   std::cout << "All real type checks passed.\n";
   return 0;
}
