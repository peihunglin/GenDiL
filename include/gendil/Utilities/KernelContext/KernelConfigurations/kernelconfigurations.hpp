// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file kernelconfigurations.hpp
 * @brief Umbrella for GenDiL's concrete kernel execution configurations.
 *
 * This narrower umbrella provides execution policies without the KernelContext
 * sizing utilities and associated traits collected by kernelconfiguration.hpp.
 */

// OpenMP-capable host execution with one logical thread per work item.
#include "host.hpp"

// Legacy device mapping with one semantic work item per physical block.
#include "threadfirst.hpp"

// Device mapping supporting independent batched work items per physical block.
#include "device.hpp"

// Backward-compatible name for HostKernelConfiguration.
#include "serial.hpp"

#ifdef GENDIL_K3_HETERO_OPENMP
// Opt-in K3-only empirical X100/A100 OpenMP execution policy.
#include "k3heterogeneousopenmp.hpp"
#endif
