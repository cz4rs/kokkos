/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <cstdio>
#include <PerfTest_Category.hpp>

namespace Test {

template <class ViewTypeA, class ViewTypeB>
double deepcopy_view(ViewTypeA& a, ViewTypeB& b, int repeat) {
  Kokkos::Timer timer;
  for (int i = 0; i < repeat; i++) {
    Kokkos::deep_copy(a, b);
  }
  Kokkos::fence();
  return timer.seconds();
}

template <class LayoutA, class LayoutB>
void run_deepcopyview_tests123(int N, int R) {
  const int N1 = N;
  const int N2 = N1 * N1;
  const int N3 = N2 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;

  double time1, time2, time3, time_raw = 100000.0;
  {
    Kokkos::View<double*, LayoutA> a("A1", N8);
    Kokkos::View<double*, LayoutB> b("B1", N8);
    time1 = deepcopy_view(a, b, R) / R;
  }
  {
    Kokkos::View<double**, LayoutA> a("A2", N4, N4);
    Kokkos::View<double**, LayoutB> b("B2", N4, N4);
    time2 = deepcopy_view(a, b, R) / R;
  }
  {
    Kokkos::View<double***, LayoutA> a("A3", N3, N3, N2);
    Kokkos::View<double***, LayoutB> b("B3", N3, N3, N2);
    time3 = deepcopy_view(a, b, R) / R;
  }
#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
  {
    Kokkos::View<double*, LayoutA> a("A1", N8);
    Kokkos::View<double*, LayoutB> b("B1", N8);
    double* const a_ptr       = a.data();
    const double* const b_ptr = b.data();
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = b_ptr[i]; });
    }
    Kokkos::fence();
    time_raw = timer.seconds() / R;
  }
#endif
  double size = 1.0 * N8 * 8 / 1024 / 1024;
  printf("   Raw:   %lf s   %lf MB   %lf GB/s\n", time_raw, size,
         2.0 * size / 1024 / time_raw);
  printf("   Rank1: %lf s   %lf MB   %lf GB/s\n", time1, size,
         2.0 * size / 1024 / time1);
  printf("   Rank2: %lf s   %lf MB   %lf GB/s\n", time2, size,
         2.0 * size / 1024 / time2);
  printf("   Rank3: %lf s   %lf MB   %lf GB/s\n", time3, size,
         2.0 * size / 1024 / time3);
}

}  // namespace Test
