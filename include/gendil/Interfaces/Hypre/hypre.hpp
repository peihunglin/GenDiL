// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

#include <type_traits>

#include "gendil/Utilities/types.hpp"

// The Hypre interface creates zero-copy views over GenDiL buffers, which
// requires gendil::Real to match HYPRE_Real (float or double).
static_assert(
   std::is_same_v< gendil::Real, double > || std::is_same_v< gendil::Real, float >,
   "The Hypre interface requires GENDIL_REAL_TYPE=DOUBLE or FLOAT." );

// Hypre backend tags and small type-level interface markers.
#include "gendil/Interfaces/Hypre/hypretypes.hpp"
// Error-checking helper for Hypre functions that return integer status codes.
#include "gendil/Interfaces/Hypre/hypreerror.hpp"
// Lightweight RAII guard for Hypre's global initialize/finalize lifecycle.
#include "gendil/Interfaces/Hypre/hypresession.hpp"
// Zero-copy `HYPRE_ParCSRMatrix` view creation for `HypreCSRMatrix`.
#include "gendil/Interfaces/Hypre/hypreparcsrview.hpp"
// Zero-copy `HYPRE_ParVector` views over compatible GenDiL vectors.
#include "gendil/Interfaces/Hypre/hyprevector.hpp"
// Hypre-backed ParCSR matrix-vector multiply entry points.
#include "gendil/Interfaces/Hypre/hyprematvec.hpp"

#endif // GENDIL_USE_HYPRE
