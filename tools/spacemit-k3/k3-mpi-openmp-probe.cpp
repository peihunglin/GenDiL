// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "k3-rvv-probe.hpp"

#include <array>
#include <cstring>
#include <cstdio>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#if !defined( __linux__ ) || !defined( __riscv )
#error "The K3 MPI/OpenMP probe requires RISC-V Linux"
#endif

namespace
{

constexpr int expected_ranks = 2;
constexpr int workers_per_rank = 8;
constexpr int stability_rounds = 64;

struct ThreadResult
{
   int rank = -1;
   int omp_thread = -1;
   long linux_thread = -1;
   int initial_cpu = -1;
   int final_cpu = -1;
   std::size_t vlen_bytes = 0;
   std::size_t vlmax_e64m1 = 0;
   unsigned long long affinity_mask = 0;
   int class_changes = 0;
   int vlen_changes = 0;
   int canary_failures = 0;
   bool passed = false;
};

bool CpuMatchesRole( const bool is_x100, const int cpu )
{
   const int first_cpu = is_x100 ? 0 : 8;
   return cpu >= first_cpu && cpu < first_cpu + workers_per_rank;
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

bool AffinityMatchesRole( const bool is_x100, const unsigned long long mask )
{
   constexpr unsigned long long x100_mask = 0x00ffULL;
   constexpr unsigned long long a100_mask = 0xff00ULL;
   const unsigned long long expected = is_x100 ? x100_mask : a100_mask;
   const bool singleton = mask != 0 && ( mask & ( mask - 1 ) ) == 0;
   return singleton && ( mask & expected ) != 0 && ( mask & ~expected ) == 0;
}

bool AffinityWithinRole( const bool is_x100, const unsigned long long mask )
{
   constexpr unsigned long long x100_mask = 0x00ffULL;
   constexpr unsigned long long a100_mask = 0xff00ULL;
   const unsigned long long expected = is_x100 ? x100_mask : a100_mask;
   return ( mask & expected ) != 0 && ( mask & ~expected ) == 0;
}

} // namespace

int main( int argc, char ** argv )
{
   const bool valid_role = argc == 2 &&
      ( std::strcmp( argv[ 1 ], "x100" ) == 0 ||
        std::strcmp( argv[ 1 ], "a100" ) == 0 );
   if ( !valid_role )
   {
      std::fprintf( stderr, "usage: %s {x100|a100}\n", argv[ 0 ] );
      return 2;
   }
   const bool is_x100 = std::strcmp( argv[ 1 ], "x100" ) == 0;
   const int pre_mpi_cpu = sched_getcpu();
   const unsigned long long pre_mpi_affinity = CurrentAffinityMask();
   if ( !CpuMatchesRole( is_x100, pre_mpi_cpu ) ||
        !AffinityWithinRole( is_x100, pre_mpi_affinity ) )
   {
      std::fprintf(
         stderr,
         "pre-MPI placement failed: role=%s cpu=%d affinity=0x%016llx\n",
         is_x100 ? "x100" : "a100",
         pre_mpi_cpu,
         pre_mpi_affinity );
      return 3;
   }

   int provided = MPI_THREAD_SINGLE;
   if ( MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided ) !=
        MPI_SUCCESS )
   {
      return 4;
   }

   int rank = -1;
   int ranks = 0;
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   MPI_Comm_size( MPI_COMM_WORLD, &ranks );

   const int post_mpi_cpu = sched_getcpu();
   const unsigned long long post_mpi_affinity = CurrentAffinityMask();
   const bool mpi_preserved_placement =
      CpuMatchesRole( is_x100, post_mpi_cpu ) &&
      AffinityWithinRole( is_x100, post_mpi_affinity ) &&
      post_mpi_affinity == pre_mpi_affinity;

   const int role_value = is_x100 ? 0 : 1;
   int role_sum = 0;
   MPI_Allreduce(
      &role_value,
      &role_sum,
      1,
      MPI_INT,
      MPI_SUM,
      MPI_COMM_WORLD );

   const int local_preflight =
      ranks == expected_ranks &&
      provided >= MPI_THREAD_FUNNELED &&
      mpi_preserved_placement &&
      role_sum == 1;
   int global_preflight = 0;
   MPI_Allreduce(
      &local_preflight,
      &global_preflight,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD );
   if ( !global_preflight )
   {
      if ( rank == 0 )
      {
         std::fprintf(
            stderr,
            "MPI preflight failed: ranks=%d thread_level=%d role_sum=%d\n",
            ranks,
            provided,
            role_sum );
      }
      MPI_Finalize();
      return 5;
   }

   std::array< ThreadResult, workers_per_rank > results{};
   int actual_workers = 0;
   omp_set_dynamic( 0 );

   #pragma omp parallel num_threads(workers_per_rank) shared(results, actual_workers)
   {
      const int thread = omp_get_thread_num();
      ThreadResult result;
      result.rank = rank;
      result.omp_thread = thread;
      result.linux_thread = syscall( SYS_gettid );
      result.initial_cpu = sched_getcpu();
      result.affinity_mask = CurrentAffinityMask();

      #pragma omp single
      actual_workers = omp_get_num_threads();

      if ( CpuMatchesRole( is_x100, result.initial_cpu ) &&
           AffinityMatchesRole( is_x100, result.affinity_mask ) )
      {
         if ( !gendil_k3_probe_rvv(
                 &result.vlen_bytes,
                 &result.vlmax_e64m1,
                 static_cast< std::uint64_t >( result.linux_thread ) ) )
         {
            ++result.canary_failures;
         }
      }
      else
      {
         ++result.class_changes;
      }

      #pragma omp barrier

      for ( int round = 0; round < stability_rounds; ++round )
      {
         const int cpu = sched_getcpu();
         if ( !CpuMatchesRole( is_x100, cpu ) )
         {
            ++result.class_changes;
         }

         if ( result.vlen_bytes != 0 )
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
      const std::size_t expected_vlen = is_x100 ? 32 : 128;
      const std::size_t expected_vlmax = is_x100 ? 4 : 16;
      result.passed =
         ranks == expected_ranks &&
         CpuMatchesRole( is_x100, result.initial_cpu ) &&
         CpuMatchesRole( is_x100, result.final_cpu ) &&
         AffinityMatchesRole( is_x100, result.affinity_mask ) &&
         result.vlen_bytes == expected_vlen &&
         result.vlmax_e64m1 == expected_vlmax &&
         result.class_changes == 0 &&
         result.vlen_changes == 0 &&
         result.canary_failures == 0;
      results[ thread ] = result;
   }

   int local_pass = actual_workers == workers_per_rank ? 1 : 0;
   unsigned long long affinity_coverage = 0;
   for ( const ThreadResult & result : results )
   {
      local_pass = local_pass && result.passed;
      affinity_coverage |= result.affinity_mask;
   }
   const unsigned long long expected_coverage = is_x100 ? 0x00ffULL : 0xff00ULL;
   local_pass = local_pass && affinity_coverage == expected_coverage;

   const int communication_value = rank + 1;
   int communication_sum = 0;
   MPI_Allreduce(
      &communication_value,
      &communication_sum,
      1,
      MPI_INT,
      MPI_SUM,
      MPI_COMM_WORLD );
   local_pass = local_pass && communication_sum == 3;

   int global_pass = 0;
   MPI_Allreduce(
      &local_pass,
      &global_pass,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD );

   for ( int output_rank = 0; output_rank < ranks; ++output_rank )
   {
      MPI_Barrier( MPI_COMM_WORLD );
      if ( rank == output_rank )
      {
         std::printf(
            "rank=%d role=%s ranks=%d mpi_thread_provided=%d "
            "pre_mpi_cpu=%d post_mpi_cpu=%d "
            "pre_mpi_affinity=0x%016llx post_mpi_affinity=0x%016llx "
            "actual_workers=%d affinity_coverage=0x%016llx "
            "communication_sum=%d\n",
            rank,
            is_x100 ? "x100" : "a100",
            ranks,
            provided,
            pre_mpi_cpu,
            post_mpi_cpu,
            pre_mpi_affinity,
            post_mpi_affinity,
            actual_workers,
            affinity_coverage,
            communication_sum );
         std::printf(
            "rank,omp_thread,linux_thread,initial_cpu,final_cpu,vlen_bytes,"
            "vlmax_e64m1,affinity_mask,class_changes,vlen_changes,"
            "canary_failures,status\n" );
         for ( const ThreadResult & result : results )
         {
            std::printf(
               "%d,%d,%ld,%d,%d,%zu,%zu,0x%016llx,%d,%d,%d,%s\n",
               result.rank,
               result.omp_thread,
               result.linux_thread,
               result.initial_cpu,
               result.final_cpu,
               result.vlen_bytes,
               result.vlmax_e64m1,
               result.affinity_mask,
               result.class_changes,
               result.vlen_changes,
               result.canary_failures,
               result.passed ? "pass" : "fail" );
         }
         std::fflush( stdout );
      }
   }

   MPI_Barrier( MPI_COMM_WORLD );
   if ( rank == 0 )
   {
      std::printf(
         "mpi_openmp_heterogeneous_probe=%s\n",
         global_pass ? "pass" : "fail" );
      std::fflush( stdout );
   }

   MPI_Finalize();
   return global_pass ? 0 : 1;
}
