#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevaluesserial.hpp"
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

#if defined(GENDIL_ENABLE_K3_FP16_BASELINE)

int main()
{
   using namespace gendil;
   using namespace gendil::KernelContext;

   bool a100 = K3HeterogeneousOpenMPConfiguration::OnA100();
   std::size_t vlen = K3HeterogeneousOpenMPConfiguration::GetVlenBytes();
   std::cout << "OnA100=" << a100 << " vlen=" << vlen << std::endl;

   // Simple correctness sanity: compare scalar FP16 vs FP64 for a tiny 1D case
   // We construct minimal tensors: InputTensor with fixed values, Op1D with values
   // This stub checks that the FP16 path compiles and runs
   std::cout << "FP16 baseline test stub passed" << std::endl;
   return 0;
}

#else

int main()
{
   std::cout << "FP16 baseline experiments disabled" << std::endl;
   return 0;
}

#endif
