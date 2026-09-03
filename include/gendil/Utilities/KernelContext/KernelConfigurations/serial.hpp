// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_K3_HETERO_OPENMP
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
#else
#include "gendil/Utilities/KernelContext/KernelConfigurations/host.hpp"
#endif

namespace gendil
{

#ifdef GENDIL_K3_HETERO_OPENMP
using SerialKernelConfiguration = K3HeterogeneousOpenMPConfiguration;
#else
using SerialKernelConfiguration = HostKernelConfiguration;
#endif

} // namespace gendil
