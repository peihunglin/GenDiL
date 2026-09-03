// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file k3heterogeneousopenmp.hpp
 * @brief Empirical SpacemiT K3 X100/A100 OpenMP execution policy.
 *
 * This policy follows the K3 GEMM reference mechanism. It is intentionally
 * opt-in and Linux/RISC-V-specific: workers 0..7 are pinned to X100 CPUs and
 * workers 8..15 are moved to the A100 class by Linux TID. It partitions each
 * work-item range by global index, so the callback receives the same index it
 * would receive under HostKernelConfiguration.
 *
 * The policy is an execution study, not a general OpenMP affinity abstraction.
 * A100 placement confines workers to the A100 class but does not guarantee one
 * worker per A100 CPU. Do not use this policy unless the K3 placement probe has
 * passed on the target system.
 */

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/KernelContext/threadlayout.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace gendil
{

class K3HeterogeneousOpenMPConfiguration
{
private:
   static constexpr GlobalIndex x100_workers = 8;
   static constexpr GlobalIndex a100_workers = 8;
   static constexpr GlobalIndex total_workers = x100_workers + a100_workers;

   static int A100Share()
   {
      const char *value = std::getenv( "GENDIL_K3_A100_SHARE" );
      if ( value == nullptr || value[ 0 ] == '\0' )
      {
         return 50;
      }
      const int share = std::atoi( value );
      if ( share < 0 || share > 100 )
      {
         std::fprintf(
            stderr,
            "GENDIL_K3_A100_SHARE must be in [0,100], got %s\n",
            value );
         std::abort();
      }
      return share;
   }

   static GlobalIndex Split( const GlobalIndex count )
   {
      const auto share = static_cast< GlobalIndex >( A100Share() );
      return count - count * share / 100;
   }

   static void Fail( const char *operation )
   {
      std::fprintf(
         stderr,
         "K3 heterogeneous OpenMP operation failed: %s (errno=%d)\n",
         operation,
         errno );
      std::abort();
   }

   static void BindX100Worker( const GlobalIndex worker )
   {
      cpu_set_t cpus;
      CPU_ZERO( &cpus );
      CPU_SET( static_cast< int >( worker ), &cpus );
      if ( sched_setaffinity( 0, sizeof( cpus ), &cpus ) != 0 )
      {
         Fail( "sched_setaffinity" );
      }
   }

   static void MoveA100Worker()
   {
      char text[ 32 ];
      const auto tid = syscall( SYS_gettid );
      const int length = std::snprintf( text, sizeof( text ), "%ld", tid );
      if ( length <= 0 || length >= static_cast< int >( sizeof( text ) ) )
      {
         errno = EOVERFLOW;
         Fail( "format A100 TID" );
      }

      const int fd = open( "/proc/set_ai_thread", O_WRONLY | O_CLOEXEC );
      if ( fd < 0 )
      {
         Fail( "open /proc/set_ai_thread" );
      }
      const auto written = write( fd, text, static_cast< std::size_t >( length ) );
      const int write_errno = errno;
      close( fd );
      if ( written != length )
      {
         errno = written < 0 ? write_errno : EIO;
         Fail( "write /proc/set_ai_thread" );
      }
   }

   static void BindWorker( const GlobalIndex worker )
   {
      if ( worker < x100_workers )
      {
         BindX100Worker( worker );
      }
      else
      {
         MoveA100Worker();
      }
   }

   static void EnsureWorkerBound( const GlobalIndex worker )
   {
      // OpenMP may reuse its pthreads across BlockLoop calls. Avoid paying the
      // K3 placement syscall cost inside every timed operator invocation, while
      // still rebinding if a runtime maps a pthread to another worker number.
      thread_local GlobalIndex bound_worker = total_workers;
      if ( bound_worker != worker )
      {
         BindWorker( worker );
         bound_worker = worker;
      }
   }

public:
   using thread_layout_type = ThreadBlockLayout<>;

   static constexpr bool is_host_configuration = true;
   static constexpr bool is_device_configuration = false;
   static constexpr size_t batch_size = 1;
   static constexpr size_t thread_block_dim = 0;
   static constexpr size_t shared_block_max_dim = 0;

   template < Integer space_dim >
   using threaded_dimensions = std::index_sequence<>;
   template < Integer space_dim >
   using register_dimensions = std::make_index_sequence< space_dim >;
   template < Integer space_dim >
   using non_shared_register_dimensions = std::make_index_sequence< space_dim >;
   template < Integer space_dim >
   using shared_register_dimensions = std::index_sequence<>;
   template < Integer space_dim >
   using shared_dimensions = std::index_sequence<>;

   static constexpr size_t GetNumberOfThreads()
   {
      return total_workers;
   }

   static inline void Synchronize()
   {
   }

   template < typename Lambda >
   static void BlockLoop( const GlobalIndex count, Lambda && body )
   {
      omp_set_dynamic( 0 );
      const GlobalIndex split = Split( count );

      #pragma omp parallel num_threads(total_workers) shared(body, split, count)
      {
         const auto worker = static_cast< GlobalIndex >( omp_get_thread_num() );
         EnsureWorkerBound( worker );
         #pragma omp barrier

         if ( worker < x100_workers )
         {
            for ( GlobalIndex index = worker; index < split; index += x100_workers )
            {
               body( index );
            }
         }
         else
         {
            const auto local_worker = worker - x100_workers;
            for (
               GlobalIndex index = split + local_worker;
               index < count;
               index += a100_workers )
            {
               body( index );
            }
         }
      }
   }
};

} // namespace gendil
