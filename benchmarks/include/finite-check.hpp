// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <gendil/MatrixFreeOperators/KernelOperators/vector.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace gendil::benchmark
{

/**
 * @brief Abort a benchmark if a vector contains a NaN or infinity.
 *
 * This is intended to run once after a timed region.  It reads device-resident
 * data back to the host when necessary, so it must not be placed in the timing
 * loop.
 */
inline void VerifyFinite( const Vector & values, std::string_view label )
{
   const Real * data = values.ReadHostData();
   for ( Integer i = 0; i < values.Size(); ++i )
   {
      if ( !std::isfinite( static_cast< double >( data[i] ) ) )
      {
         std::cerr << "Non-finite value in " << label << " at DoF " << i
                   << ": " << static_cast< double >( data[i] ) << '\n';
         std::exit( EXIT_FAILURE );
      }
   }
}

} // namespace gendil::benchmark
