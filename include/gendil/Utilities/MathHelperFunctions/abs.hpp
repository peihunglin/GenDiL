// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

#include <cmath>

namespace gendil {

template < typename T >
GENDIL_HOST_DEVICE
auto Abs( const T & val )
{
#if GENDIL_LOW_PRECISION_REAL
   if constexpr ( std::is_same_v< std::remove_cv_t< T >, Real > )
   {
      // Low precision types have no device fabs; compare against zero instead.
      return val < Real{ 0 } ? static_cast< Real >( -val ) : static_cast< Real >( val );
   }
   else
#endif
   {
      return std::fabs( val );
   }
}

}
