#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevaluesserial.hpp"
#include "gendil/Utilities/KernelContext/KernelConfigurations/k3heterogeneousopenmp.hpp"
#include <iostream>
#include <cassert>

#if defined(GENDIL_ENABLE_K3_IME_EXPERIMENTS)

int main()
{
   using namespace gendil;
   using namespace gendil::details;
   using namespace gendil::KernelContext;

   // Simple sanity: ensure OnA100() is callable
   bool a100 = K3HeterogeneousOpenMPConfiguration::OnA100();
   std::cout << "OnA100=" << a100 << " vlen=" << K3HeterogeneousOpenMPConfiguration::GetVlenBytes() << std::endl;

   // Placeholder for real correctness test
   std::cout << "IME pilot test stub passed" << std::endl;
   return 0;
}

#else

int main()
{
   std::cout << "IME experiments disabled" << std::endl;
   return 0;
}

#endif
