// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstddef>
#include <cstdint>

// Implemented in an RVV-only translation unit. Keeping this declaration free
// of RVV types lets the coordinator compile for rv64gc.
extern "C" int gendil_k3_probe_rvv(
   std::size_t * vlen_bytes,
   std::size_t * vlmax_e64m1,
   std::uint64_t seed );
