// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstddef>
#include <cstdlib>
#include <utility>
#include <array>
#include <iostream>
#include <tuple>
#include <concepts>
#include <functional>
#ifdef GENDIL_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#ifdef GENDIL_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace gendil {

// Type used for local indices
using LocalIndex = int;

// Type used for global indices
#if GENDIL_USE_DEVICE
using GlobalIndex = size_t;
#else
using GlobalIndex = size_t;
#endif

// Type used for positive integers
#if GENDIL_USE_DEVICE
using Integer = size_t;
#else
using Integer = size_t;
#endif

// Type used for real numbers.
//
// Selected at configure time through the GENDIL_REAL_TYPE CMake option, which
// defines exactly one of GENDIL_REAL_TYPE_DOUBLE (default), GENDIL_REAL_TYPE_FLOAT,
// GENDIL_REAL_TYPE_FP16, or GENDIL_REAL_TYPE_BF16. The FP16 and BF16 modes map
// Real onto the compiler's native extended floating-point types so that the
// whole library, including the tests, runs in purely low precision arithmetic.
#if defined( GENDIL_REAL_TYPE_FP16 )
#if defined( __CUDACC__ ) && !defined( __clang__ )
#error "GENDIL_REAL_TYPE_FP16 requires Clang as the CUDA compiler: NVCC does not support _Float16 arithmetic."
#endif
using Real = _Float16;
#define GENDIL_LOW_PRECISION_REAL 1
#elif defined( GENDIL_REAL_TYPE_BF16 )
#if defined( __CUDACC__ ) && !defined( __clang__ )
#error "GENDIL_REAL_TYPE_BF16 requires Clang as the CUDA compiler: NVCC does not support __bf16 arithmetic."
#endif
using Real = __bf16;
#define GENDIL_LOW_PRECISION_REAL 1
#elif defined( GENDIL_REAL_TYPE_FLOAT )
using Real = float;
#define GENDIL_LOW_PRECISION_REAL 0
#else
using Real = double;
#define GENDIL_LOW_PRECISION_REAL 0
#endif

struct Empty { };

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GENDIL_DEVICE_CODE
#endif

#ifndef _GENDIL_FUNC_NAME
#ifndef _MSC_VER
#define _GENDIL_FUNC_NAME __PRETTY_FUNCTION__
#else
// for Visual Studio C++
#define _GENDIL_FUNC_NAME __FUNCSIG__
#endif
#endif

// CUDA
#if defined( GENDIL_USE_CUDA )

using Stream_t = cudaStream_t;

#define GENDIL_HOST_DEVICE __host__ __device__

void cuda_error(
   cudaError_t error,
   const char *message,
   const char *func,
   const char *file,
   int line )
{
   std::cerr << "\n\nCUDA error: (" << cudaGetErrorString(error) << ") failed with message:\n"
             << message
             << "\n ... in file: " << file << ':' << line
             << "\n ... in function: " << func << std::endl;
   std::abort();
}

#define GENDIL_DEVICE_SYNC GENDIL_GPU_CHECK(cudaDeviceSynchronize())
#define GENDIL_STREAM_SYNC GENDIL_GPU_CHECK(cudaStreamSynchronize(0))
#define GENDIL_GPU_CHECK(x) \
   do \
   { \
      cudaError_t err = (x); \
      if (err != cudaSuccess) \
      { \
         cuda_error(err, #x, _GENDIL_FUNC_NAME, __FILE__, __LINE__); \
      } \
   } \
   while (0)

// HIP
#elif defined( GENDIL_USE_HIP )

using Stream_t = hipStream_t;

void hip_error(
   hipError_t error,
   const char *message,
   const char *func,
   const char *file,
   int line )
{
   std::cerr << "\n\nHIP error: (" << hipGetErrorString(error) << ") failed with message:\n"
             << message
             << "\n ... in file: " << file << ':' << line
             << "\n ... in function: " << func << std::endl;
   std::abort();
}

#define GENDIL_HOST_DEVICE __host__ __device__
#define GENDIL_DEVICE_SYNC GENDIL_GPU_CHECK(hipDeviceSynchronize())
#define GENDIL_STREAM_SYNC GENDIL_GPU_CHECK(hipStreamSynchronize(0))
#define GENDIL_GPU_CHECK(x) \
   do \
   { \
      hipError_t err = (x); \
      if (err != hipSuccess) \
      { \
         hip_error(err, #x, _GENDIL_FUNC_NAME, __FILE__, __LINE__); \
      } \
   } \
   while (0)

#else

#define GENDIL_HOST_DEVICE
#define GENDIL_DEVICE_SYNC 
#define GENDIL_STREAM_SYNC
#define GENDIL_GPU_CHECK(x) x

#endif

#if GENDIL_USE_DEVICE
#define GENDIL_DEVICE __device__
#else
#define GENDIL_DEVICE
#endif

#ifdef GENDIL_DEVICE_CODE
#define GENDIL_SHARED __shared__
#else
#define GENDIL_SHARED
#endif

#ifdef GENDIL_DEVICE_CODE
#define GENDIL_SYNC_THREADS() __syncthreads()
#else
#define GENDIL_SYNC_THREADS()
#endif

#if defined( GENDIL_USE_CUDA )
#define GENDIL_INLINE __forceinline__
#elif defined( GENDIL_USE_HIP )
#define GENDIL_INLINE inline  __attribute__((always_inline))
#elif defined (_WIN32)
#define GENDIL_INLINE inline
#else
#define GENDIL_INLINE inline  __attribute__((always_inline))
#endif

}

// Standard library support for the low precision real types and the
// precision-aware helpers (RealTolerance, RealTypeName, ...).
#include "gendil/Utilities/realtype.hpp"
