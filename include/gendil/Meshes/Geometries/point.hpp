// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <ostream>
#include <type_traits>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/sqrt.hpp"

namespace gendil {

/**
 * @brief A structure describing a point in an arbitrary dimension space.
 * 
 * @tparam Dim The dimension of the space embedding the point.
 */
template < size_t Dim >
struct Point
{
   static constexpr size_t space_dim = Dim;
   static constexpr size_t geometry_dim = 0;
   static constexpr size_t num_nodes = 1;
   static constexpr size_t num_faces = 0;

   Real coordinates [ Dim ];

   constexpr Point() : coordinates{} { }

   template < typename... Args >
   constexpr Point( Args... args ) : coordinates{ static_cast< Real >( args )... }
   {
      static_assert(
         sizeof...(Args) == Dim,
         "Wrong number of coordinates"
      );
   }

   constexpr Real operator[]( size_t index ) const
   {
      // TODO: Add debug assertion?
      return coordinates[ index ];
   }

   constexpr Real& operator[]( size_t index )
   {
      // TODO: Add debug assertion?
      return coordinates[ index ];
   }
};

// Specialize the scalar case
template <>
struct Point<1>
{
   static constexpr size_t space_dim = 1;
   static constexpr size_t geometry_dim = 0;
   static constexpr size_t num_nodes = 1;
   static constexpr size_t num_faces = 0;

   Real coordinates [ space_dim ];

   constexpr Point() : coordinates{} { }

   // Automatically cast a Real into a Point<1>.
   constexpr Point( Real arg ) : coordinates{ arg }
   { }

   // Automatically cast a Point<1> into a Real.
   constexpr operator Real() const
   {
      return coordinates[ 0 ];
   }

   constexpr Real operator[]( size_t index ) const
   {
      // TODO: Add debug assertion?
      return coordinates[ 0 ];
   }

   constexpr Real& operator[]( size_t index )
   {
      // TODO: Add debug assertion?
      return coordinates[ 0 ];
   }
};

namespace internal {

/// Scalar types that may combine with a Point<1> in arithmetic expressions.
template < typename T >
concept PointScalar =
   std::is_arithmetic_v< std::remove_cvref_t< T > > ||
   std::is_same_v< std::remove_cvref_t< T >, Real >;

}

// Arithmetic between the scalar point and a scalar. Point<1> converts
// implicitly to Real, but the built-in operator candidates do not cover the
// low precision real types, so the mixed cases are spelled out explicitly;
// they evaluate exactly as the implicit conversion would.
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator+( Point<1> const & a, const T & b ) { return a[0] + b; }
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator+( const T & a, Point<1> const & b ) { return a + b[0]; }
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator-( Point<1> const & a, const T & b ) { return a[0] - b; }
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator-( const T & a, Point<1> const & b ) { return a - b[0]; }
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator*( Point<1> const & a, const T & b ) { return a[0] * b; }
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator*( const T & a, Point<1> const & b ) { return a * b[0]; }
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator/( Point<1> const & a, const T & b ) { return a[0] / b; }
template < internal::PointScalar T >
GENDIL_HOST_DEVICE constexpr auto operator/( const T & a, Point<1> const & b ) { return a / b[0]; }

template < size_t Dim >
std::ostream& operator<<(std::ostream& os, Point<Dim> const & index) {
   os << "( ";
   for (size_t i = 0; i < Dim; i++)
   {
      os << index[ i ] << " ";
   }
   os << ")";
   return os;
}

template < size_t Dim >
bool operator==( Point<Dim> const & a, Point<Dim> const & b )
{
   bool equal = true;
   for ( size_t i = 0; i < Dim; ++i )
   {
      equal = equal && (a[i] == b[i]);
   }
   return equal;
}

template < size_t Dim >
auto operator+( Point<Dim> const & a, Point<Dim> const & b )
{
   Point<Dim> result;
   for ( size_t i = 0; i < Dim; ++i )
   {
      result[i] = a[i] + b[i];
   }
   return result;
}

template < size_t Dim >
auto operator-( Point<Dim> const & a, Point<Dim> const & b )
{
   Point<Dim> result;
   for ( size_t i = 0; i < Dim; ++i )
   {
      result[i] = a[i] - b[i];
   }
   return result;
}

template < size_t Dim >
auto operator*( std::array<Real,Dim> const & a, Point<Dim> const & b )
{
   Point<Dim> result;
   for ( size_t i = 0; i < Dim; ++i )
   {
      result[i] = a[i] * b[i];
   }
   return result;
}

template < size_t Dim >
Real Distance( Point<Dim> const & a, Point<Dim> const & b )
{
   Real result = Real{ 0 };
   for ( size_t i = 0; i < Dim; ++i )
   {
      Real d = a[i] - b[i];
      result += d * d;
   }
   return Sqrt(result);
}

template < size_t Dim >
bool AlmostEqual( Point<Dim> const & a, Point<Dim> const & b )
{
   const double tol = RealTolerance( 1e-8 );
   return Distance(a, b) < tol;
}

}
