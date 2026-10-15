// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Verifies the standard library support and precision helpers provided for
// gendil::Real, in particular when the library is configured with one of the
// low precision real types (GENDIL_REAL_TYPE=FP16 or BF16).

#include <gendil/gendil.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
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
   using limits = std::numeric_limits< Real >;
   Check( limits::is_specialized, "numeric_limits<Real> is specialized" );
   Check( limits::is_iec559, "numeric_limits<Real> is IEEE" );
   Check( limits::digits == real_significand_digits_v, "significand digits exposed" );

   // 1 + epsilon is the smallest representable number above one.
   const Real one = Real( 1 );
   const Real eps = limits::epsilon();
   const Real half_eps = static_cast< Real >( static_cast< double >( eps ) * 0.5 );
   Check( one + eps > one, "1 + epsilon > 1" );
   Check( one + half_eps == one, "1 + epsilon / 2 == 1" );

   // The reported extrema are finite and bracket the representable range.
   Check( std::isfinite( limits::max() ), "max is finite" );
   Check( std::isinf( limits::infinity() ), "infinity is infinite" );
   Check( std::isnan( limits::quiet_NaN() ), "quiet_NaN is NaN" );
   Check( limits::lowest() == -limits::max(), "lowest == -max" );
   Check( limits::min() > Real( 0 ), "min is positive" );
   Check( limits::denorm_min() > Real( 0 ), "denorm_min is positive" );
   Check( limits::denorm_min() < limits::min(), "denorm_min < min" );

   if constexpr ( is_low_precision_real_v )
   {
      // Doubling the maximum overflows, halving denorm_min underflows.
      const Real two_max = static_cast< Real >( static_cast< float >( limits::max() ) * 2.0f );
      Check( std::isinf( two_max ), "2 * max overflows" );
      const Real half_denorm = static_cast< Real >( static_cast< float >( limits::denorm_min() ) * 0.5f );
      Check( half_denorm == Real( 0 ), "denorm_min / 2 underflows" );
      Check( sizeof( Real ) == 2, "low precision Real is 16 bits wide" );
   }
   return failures == 0;
}

bool TestMathFunctions()
{
   const Real x = Real( 2.25 );
   Check( std::sqrt( x ) == Real( 1.5 ), "std::sqrt" );
   Check( Sqrt( x ) == Real( 1.5 ), "gendil::Sqrt" );
   Check( std::abs( Real( -1.5 ) ) == Real( 1.5 ), "std::abs" );
   Check( std::fabs( Real( -1.5 ) ) == Real( 1.5 ), "std::fabs" );
   Check( Abs( Real( -1.5 ) ) == Real( 1.5 ), "gendil::Abs" );
   Check( std::pow( Real( 2 ), Real( 3 ) ) == Real( 8 ), "std::pow(Real, Real)" );
   Check( std::pow( Real( 2 ), 3 ) == Real( 8 ), "std::pow(Real, int)" );
   Check( std::pow( Real( 2 ), 3.0 ) == 8.0, "std::pow(Real, double)" );
   Check( Pow< 3 >( Real( 2 ) ) == Real( 8 ), "gendil::Pow" );
   Check( std::floor( Real( 1.75 ) ) == Real( 1 ), "std::floor" );
   Check( std::ceil( Real( 1.25 ) ) == Real( 2 ), "std::ceil" );
   Check( std::isfinite( x ), "std::isfinite" );
   Check( !std::isnan( x ), "std::isnan" );
   Check( std::max( x, Real( 1 ) ) == x, "std::max" );
   Check( std::min( x, Real( 1 ) ) == Real( 1 ), "std::min" );
   Check( Max( x, Real( 1 ) ) == x, "gendil::Max" );
   Check( Min( x, Real( 1 ) ) == Real( 1 ), "gendil::Min" );
   const double cos_zero = static_cast< double >( std::cos( Real( 0 ) ) );
   Check( cos_zero == 1.0, "std::cos" );
   return failures == 0;
}

bool TestStreams()
{
   std::ostringstream out;
   out << Real( 1.5 ) << ' ' << Real( -2 );
   Check( out.str() == "1.5 -2", "ostream insertion prints Real: '" + out.str() + "'" );

   std::istringstream in( "0.25" );
   Real value = Real( 0 );
   in >> value;
   Check( value == Real( 0.25 ), "istream extraction reads Real" );
   return failures == 0;
}

bool TestRealTolerance()
{
   // RealTolerance preserves the fraction of available digits that a double
   // precision tolerance demands, and is the identity in double precision.
   const double exponent =
      static_cast< double >( std::numeric_limits< Real >::digits ) /
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
   Check( RealEpsilon() == static_cast< double >( std::numeric_limits< Real >::epsilon() ), "RealEpsilon" );
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
   Check( std::isfinite( Norml2( v ) ), "Norml2 of a random vector is finite" );
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
   TestStreams();
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
