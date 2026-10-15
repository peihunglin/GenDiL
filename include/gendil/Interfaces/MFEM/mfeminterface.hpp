// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Utilities/types.hpp"

#ifdef GENDIL_USE_MFEM
// MFEM stores its scalars as double; the interface shares buffers between
// gendil::Vector and mfem::Vector and therefore requires the same real type.
static_assert(
   std::is_same_v< gendil::Real, double >,
   "The MFEM interface requires GENDIL_REAL_TYPE=DOUBLE." );
#endif

// Factories to build gendil unstructured meshes from MFEM meshes.
#include "meshfactories.hpp"

// Global face connectivity bundle types and builder declarations for MFEM meshes.
#include "GlobalFaceConnectivity/globalfaceconnectivity.hpp"

// Global-facet partition factory for MFEM meshes.
#include "globalpartition.hpp"

// An mfem::Vector wrapper to interface between gendil::Vector and mfem::Vector.
#include "vector.hpp"
