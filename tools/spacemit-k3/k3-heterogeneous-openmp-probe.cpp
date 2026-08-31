// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "k3-rvv-probe.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/prctl.h>
#include <omp.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#if !defined( __linux__ ) || !defined( __riscv )
#error "The heterogeneous OpenMP probe requires RISC-V Linux"
#endif

#if !defined( PR_RISCV_V_SET_CONTROL )
#error "Linux headers do not provide RISC-V vector-state control"
#endif

namespace
{

constexpr int num_x100_workers = 8;
constexpr int num_a100_workers = 8;
constexpr int expected_workers = num_x100_workers + num_a100_workers;
constexpr int x100_cpu_begin = 0;
constexpr int a100_cpu_begin = 8;
constexpr int stability_rounds = 64;

struct ThreadResult
{
   int omp_thread = -1;
   long linux_thread = -1;
   int initial_cpu = -1;
   int placed_cpu = -1;
   int final_cpu = -1;
   int placement_result = -1;
   int placement_errno = 0;
   long vector_control_before = -1;
   int vector_enable_result = -1;
   int vector_enable_errno = 0;
   std::size_t vlen_bytes = 0;
   std::size_t vlmax_e64m1 = 0;
   unsigned long long affinity_mask = 0;
   int class_changes = 0;
   int vlen_changes = 0;
   int canary_failures = 0;
   bool passed = false;
};

bool IsX100Cpu( const int cpu )
{
   return cpu >= x100_cpu_begin && cpu < x100_cpu_begin + num_x100_workers;
}

bool IsA100Cpu( const int cpu )
{
   return cpu >= a100_cpu_begin && cpu < a100_cpu_begin + num_a100_workers;
}

bool AffinityMatchesClass(
   const unsigned long long mask,
   const bool is_x100,
   const int omp_thread )
{
   constexpr unsigned long long x100_mask = 0x00ffULL;
   constexpr unsigned long long a100_mask = 0xff00ULL;
   if ( is_x100 )
   {
      return mask == ( 1ULL << ( x100_cpu_begin + omp_thread ) );
   }
   return ( mask & a100_mask ) != 0 && ( mask & ~a100_mask ) == 0;
}

unsigned long long CurrentAffinityMask()
{
   cpu_set_t cpus;
   CPU_ZERO( &cpus );
   if ( sched_getaffinity( 0, sizeof( cpus ), &cpus ) != 0 )
   {
      return 0;
   }

   unsigned long long mask = 0;
   for ( int cpu = 0; cpu < 64 && cpu < CPU_SETSIZE; ++cpu )
   {
      if ( CPU_ISSET( cpu, &cpus ) )
      {
         mask |= 1ULL << cpu;
      }
   }
   return mask;
}

int PinCurrentThread( const int cpu )
{
   cpu_set_t cpus;
   CPU_ZERO( &cpus );
   CPU_SET( cpu, &cpus );
   return sched_setaffinity( 0, sizeof( cpus ), &cpus );
}

int MoveCurrentThreadToA100( const long thread_id )
{
   char thread_text[ 32 ];
   const int length = std::snprintf(
      thread_text,
      sizeof( thread_text ),
      "%ld",
      thread_id );
   if ( length <= 0 || static_cast< std::size_t >( length ) >= sizeof( thread_text ) )
   {
      errno = EOVERFLOW;
      return -1;
   }

   const int fd = open( "/proc/set_ai_thread", O_WRONLY | O_CLOEXEC );
   if ( fd < 0 )
   {
      return -1;
   }

   const ssize_t written = write(
      fd,
      thread_text,
      static_cast< std::size_t >( length ) );
   const int write_errno = errno;
   close( fd );
   if ( written != length )
   {
      errno = written < 0 ? write_errno : EIO;
      return -1;
   }
   return 0;
}

int EnableCurrentThreadRvv()
{
   constexpr unsigned long next_off =
      PR_RISCV_V_VSTATE_CTRL_OFF << 2;
   constexpr unsigned long control =
      PR_RISCV_V_VSTATE_CTRL_ON |
      next_off |
      PR_RISCV_V_VSTATE_CTRL_INHERIT;
   return prctl( PR_RISCV_V_SET_CONTROL, control );
}

} // namespace

int main()
{
   std::array< ThreadResult, expected_workers > results{};
   int actual_workers = 0;

   omp_set_dynamic( 0 );

   #pragma omp parallel num_threads(expected_workers) shared(results, actual_workers)
   {
      const int thread = omp_get_thread_num();
      ThreadResult result;
      result.omp_thread = thread;
      result.linux_thread = syscall( SYS_gettid );
      result.initial_cpu = sched_getcpu();

      errno = 0;
      if ( thread < num_x100_workers )
      {
         result.placement_result = PinCurrentThread( x100_cpu_begin + thread );
      }
      else
      {
         result.placement_result = MoveCurrentThreadToA100( result.linux_thread );
      }
      result.placement_errno = result.placement_result == 0 ? 0 : errno;

      // set_ai_thread changes scheduler placement in the kernel. Give the
      // scheduler a bounded opportunity to run the worker on its new class,
      // still with RVV disabled.
      if ( result.placement_result == 0 )
      {
         for ( int attempt = 0; attempt < 10000; ++attempt )
         {
            const int cpu = sched_getcpu();
            const bool placed = thread < num_x100_workers ?
               IsX100Cpu( cpu ) : IsA100Cpu( cpu );
            if ( placed )
            {
               break;
            }
            sched_yield();
         }
      }

      // The barrier deliberately occurs while RVV remains disabled. It checks
      // that OpenMP synchronization itself does not require vector execution.
      #pragma omp barrier

      #pragma omp single
      actual_workers = omp_get_num_threads();

      result.placed_cpu = sched_getcpu();
      result.affinity_mask = CurrentAffinityMask();
      result.vector_control_before = prctl( PR_RISCV_V_GET_CONTROL );

      const bool is_x100 = thread < num_x100_workers;
      const bool placement_is_confirmed =
         result.placement_result == 0 &&
         ( is_x100 ? IsX100Cpu( result.placed_cpu ) :
                     IsA100Cpu( result.placed_cpu ) ) &&
         AffinityMatchesClass( result.affinity_mask, is_x100, thread );
      const bool rvv_is_disabled =
         result.vector_control_before >= 0 &&
         ( result.vector_control_before & PR_RISCV_V_VSTATE_CTRL_CUR_MASK ) ==
            PR_RISCV_V_VSTATE_CTRL_OFF;

      // Never establish vector state on an unconfirmed core class. A worker
      // that fails placement still participates in scalar barriers so every
      // thread can report a clean failure without deadlocking the team.
      if ( placement_is_confirmed && rvv_is_disabled )
      {
         errno = 0;
         result.vector_enable_result = EnableCurrentThreadRvv();
         result.vector_enable_errno =
            result.vector_enable_result == 0 ? 0 : errno;
         if ( result.vector_enable_result == 0 &&
              !gendil_k3_probe_rvv(
                 &result.vlen_bytes,
                 &result.vlmax_e64m1,
                 static_cast< std::uint64_t >( result.linux_thread ) ) )
         {
            ++result.canary_failures;
         }
      }
      else
      {
         result.vector_enable_result = -2;
      }

      #pragma omp barrier

      for ( int round = 0; round < stability_rounds; ++round )
      {
         const int cpu = sched_getcpu();
         const bool expected_class = thread < num_x100_workers ?
            IsX100Cpu( cpu ) : IsA100Cpu( cpu );
         if ( !expected_class )
         {
            ++result.class_changes;
         }

         if ( result.vector_enable_result == 0 )
         {
            std::size_t current_vlen = 0;
            std::size_t current_vlmax = 0;
            const std::uint64_t seed =
               static_cast< std::uint64_t >( result.linux_thread ) ^
               static_cast< std::uint64_t >( round + 1 );
            if ( !gendil_k3_probe_rvv(
                    &current_vlen,
                    &current_vlmax,
                    seed ) )
            {
               ++result.canary_failures;
            }
            if ( current_vlen != result.vlen_bytes ||
                 current_vlmax != result.vlmax_e64m1 )
            {
               ++result.vlen_changes;
            }
         }
         #pragma omp barrier
      }

      result.final_cpu = sched_getcpu();
      const bool cpu_is_correct = is_x100 ?
         IsX100Cpu( result.placed_cpu ) && IsX100Cpu( result.final_cpu ) :
         IsA100Cpu( result.placed_cpu ) && IsA100Cpu( result.final_cpu );
      const std::size_t expected_vlen = is_x100 ? 32 : 128;
      const std::size_t expected_vlmax = is_x100 ? 4 : 16;
      const bool affinity_is_correct = AffinityMatchesClass(
         result.affinity_mask,
         is_x100,
         thread );
      result.passed =
         result.placement_result == 0 &&
         result.vector_enable_result == 0 &&
         rvv_is_disabled &&
         cpu_is_correct &&
         affinity_is_correct &&
         result.vlen_bytes == expected_vlen &&
         result.vlmax_e64m1 == expected_vlmax &&
         result.class_changes == 0 &&
         result.vlen_changes == 0 &&
         result.canary_failures == 0;

      if ( thread >= 0 && thread < expected_workers )
      {
         results[ thread ] = result;
      }
   }

   std::printf(
      "actual_workers=%d expected_workers=%d stability_rounds=%d\n",
      actual_workers,
      expected_workers,
      stability_rounds );
   std::printf(
      "omp_thread,linux_thread,initial_cpu,placed_cpu,final_cpu,"
      "placement_result,placement_errno,vector_control_before,"
      "vector_enable_result,vector_enable_errno,vlen_bytes,vlmax_e64m1,"
      "affinity_mask,class_changes,vlen_changes,canary_failures,status\n" );

   bool passed = actual_workers == expected_workers;
   for ( const ThreadResult & result : results )
   {
      std::printf(
         "%d,%ld,%d,%d,%d,%d,%d,%ld,%d,%d,%zu,%zu,0x%016llx,%d,%d,%d,%s\n",
         result.omp_thread,
         result.linux_thread,
         result.initial_cpu,
         result.placed_cpu,
         result.final_cpu,
         result.placement_result,
         result.placement_errno,
         result.vector_control_before,
         result.vector_enable_result,
         result.vector_enable_errno,
         result.vlen_bytes,
         result.vlmax_e64m1,
         result.affinity_mask,
         result.class_changes,
         result.vlen_changes,
         result.canary_failures,
         result.passed ? "pass" : "fail" );
      passed = passed && result.passed;
   }

   std::printf( "heterogeneous_openmp_probe=%s\n", passed ? "pass" : "fail" );
   return passed ? 0 : 1;
}
