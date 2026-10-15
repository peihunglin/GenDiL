// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

#include <cmath>

namespace gendil {

GENDIL_HOST_DEVICE inline
Real Sqrt( Real val )
{
#if GENDIL_LOW_PRECISION_REAL
   // Device toolchains provide no half precision sqrt; evaluate in single
   // precision and round the result back to Real.
   return static_cast< Real >( std::sqrt( static_cast< float >( val ) ) );
#else
   return std::sqrt( val );
#endif
}

}
