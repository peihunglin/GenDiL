// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <system_error>

namespace gendil::benchmarks
{

inline std::uint64_t ReadRangeBenchmarkEnvironment(
   const char * name,
   const std::uint64_t default_value )
{
   const char * text = std::getenv( name );
   if ( text == nullptr || text[ 0 ] == '\0' )
   {
      return default_value;
   }

   std::uint64_t value = 0;
   const char * end = text + std::strlen( text );
   const auto result = std::from_chars( text, end, value );
   if ( result.ec != std::errc{} || result.ptr != end || value == 0 )
   {
      std::cerr << "Invalid positive integer in " << name << ": " << text
                << std::endl;
      std::abort();
   }
   return value;
}

inline std::uint64_t RangeBenchmarkMaxDofs(
   const std::uint64_t default_value )
{
   return ReadRangeBenchmarkEnvironment(
      "GENDIL_BENCHMARK_MAX_DOFS",
      default_value );
}

inline int RangeBenchmarkIterations( const int default_value )
{
   const auto value = ReadRangeBenchmarkEnvironment(
      "GENDIL_BENCHMARK_ITERATIONS",
      static_cast< std::uint64_t >( default_value ) );
   if ( value > static_cast< std::uint64_t >( std::numeric_limits< int >::max() ) )
   {
      std::cerr << "GENDIL_BENCHMARK_ITERATIONS exceeds int range: " << value
                << std::endl;
      std::abort();
   }
   return static_cast< int >( value );
}

} // namespace gendil::benchmarks
