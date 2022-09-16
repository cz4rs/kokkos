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
#include <benchmark/benchmark.h>
#include <cmath>
#include <Benchmark_Context.hpp>

namespace Test {

void report_results_resize(benchmark::State& state, double time) {
  state.SetIterationTime(time);
  const auto N8   = std::pow(state.range(0), 8);
  const auto size = N8 * 8 / 1024 / 1024;

  state.counters["MB"] = benchmark::Counter(size, benchmark::Counter::kDefaults,
                                            benchmark::Counter::OneK::kIs1024);
  state.counters[benchmark_fom("GB/s")] = benchmark::Counter(
      2.0 * size / 1024 / time, benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1024);
}

template <class Layout>
static void ViewResize_Rank1(benchmark::State& state) {
  const int N8 = std::pow(state.range(0), 8);
  Kokkos::View<double*, Layout> a("A1", N8);
  Kokkos::View<double*, Layout> a_(a);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::resize(a_, int(N8 * 1.1));
    Kokkos::fence();
    report_results_resize(state, timer.seconds());
  }
}

template <class Layout>
static void ViewResize_Rank2(benchmark::State& state) {
  const int N4 = std::pow(state.range(0), 4);
  Kokkos::View<double**, Layout> a("A2", N4, N4);
  Kokkos::View<double**, Layout> a_(a);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::resize(a_, int(N4 * 1.1), N4);
    Kokkos::fence();
    report_results_resize(state, timer.seconds());
  }
}

template <class Layout>
static void ViewResize_Rank3(benchmark::State& state) {
  const int N2 = std::pow(state.range(0), 2);
  const int N3 = std::pow(state.range(0), 3);
  Kokkos::View<double***, Layout> a("A3", N3, N3, N2);
  Kokkos::View<double***, Layout> a_(a);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::resize(a_, int(N3 * 1.1), N3, N2);
    Kokkos::fence();
    report_results_resize(state, timer.seconds());
  }
}

template <class Layout>
static void ViewResize_NoInit_Rank1(benchmark::State& state) {
  const int N8 = std::pow(state.range(0), 8);
  Kokkos::View<double*, Layout> a("A1", N8);
  Kokkos::View<double*, Layout> a_(a);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N8 * 1.1));
    Kokkos::fence();
    report_results_resize(state, timer.seconds());
  }
}

template <class Layout>
static void ViewResize_NoInit_Rank2(benchmark::State& state) {
  const int N4 = std::pow(state.range(0), 4);
  Kokkos::View<double**, Layout> a("A2", N4, N4);
  Kokkos::View<double**, Layout> a_(a);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N4 * 1.1), N4);
    Kokkos::fence();
    report_results_resize(state, timer.seconds());
  }
}

template <class Layout>
static void ViewResize_NoInit_Rank3(benchmark::State& state) {
  const int N2 = std::pow(state.range(0), 2);
  const int N3 = std::pow(state.range(0), 3);
  Kokkos::View<double***, Layout> a("A3", N3, N3, N2);
  Kokkos::View<double***, Layout> a_(a);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N3 * 1.1), N3, N2);
    Kokkos::fence();
    report_results_resize(state, timer.seconds());
  }
}

template <class Layout>
static void ViewResize_NoInit_Raw(benchmark::State& state) {
  const int N8 = std::pow(state.range(0), 8);
  Kokkos::View<double*, Layout> a("A1", N8);
  double* a_ptr = a.data();

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::View<double*, Layout> a1(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "A1"), int(N8 * 1.1));
    double* a1_ptr = a1.data();
    Kokkos::parallel_for(
        N8, KOKKOS_LAMBDA(const int& i) { a1_ptr[i] = a_ptr[i]; });
    Kokkos::fence();
    report_results_resize(state, timer.seconds());
  }
}

// template <class Layout>
// void run_resizeview_tests45(int N, int R) {
//   const int N1 = N;
//   const int N2 = N1 * N1;
//   const int N4 = N2 * N2;
//   const int N8 = N4 * N4;

//   double time4, time5, time_raw = 100000.0;
//   double time4_noinit, time5_noinit;
//   {
//     Kokkos::View<double****, Layout> a("A4", N2, N2, N2, N2);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double****, Layout> a_(a);
//       Kokkos::resize(a_, int(N2 * 1.1), N2, N2, N2);
//     }
//     time4 = timer.seconds() / R;
//   }
//   {
//     Kokkos::View<double*****, Layout> a("A5", N2, N2, N1, N1, N2);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double*****, Layout> a_(a);
//       Kokkos::resize(a_, int(N2 * 1.1), N2, N1, N1, N2);
//     }
//     time5 = timer.seconds() / R;
//   }
//   {
//     Kokkos::View<double****, Layout> a("A4", N2, N2, N2, N2);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double****, Layout> a_(a);
//       Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N2 * 1.1), N2, N2,
//                      N2);
//     }
//     time4_noinit = timer.seconds() / R;
//   }
//   {
//     Kokkos::View<double*****, Layout> a("A5", N2, N2, N1, N1, N2);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double*****, Layout> a_(a);
//       Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N2 * 1.1), N2, N1,
//       N1,
//                      N2);
//     }
//     time5_noinit = timer.seconds() / R;
//   }
// }

// template <class Layout>
// void run_resizeview_tests6(int N, int R) {
//   const int N1 = N;
//   const int N2 = N1 * N1;
//   const int N4 = N2 * N2;
//   const int N8 = N4 * N4;

//   double time6, time6_noinit, time_raw = 100000.0;
//   {
//     Kokkos::View<double******, Layout> a("A6", N2, N1, N1, N1, N1, N2);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double******, Layout> a_(a);
//       Kokkos::resize(a_, int(N2 * 1.1), N1, N1, N1, N1, N2);
//     }
//     time6 = timer.seconds() / R;
//   }
//   {
//     Kokkos::View<double******, Layout> a("A6", N2, N1, N1, N1, N1, N2);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double******, Layout> a_(a);
//       Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N2 * 1.1), N1, N1,
//       N1,
//                      N1, N2);
//     }
//     time6_noinit = timer.seconds() / R;
//   }
// }

// template <class Layout>
// void run_resizeview_tests7(int N, int R) {
//   const int N1 = N;
//   const int N2 = N1 * N1;
//   const int N4 = N2 * N2;
//   const int N8 = N4 * N4;

//   double time7, time7_noinit, time_raw = 100000.0;
//   {
//     Kokkos::View<double*******, Layout> a("A7", N2, N1, N1, N1, N1, N1, N1);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double*******, Layout> a_(a);
//       Kokkos::resize(a_, int(N2 * 1.1), N1, N1, N1, N1, N1, N1);
//     }
//     time7 = timer.seconds() / R;
//   }
//   {
//     Kokkos::View<double*******, Layout> a("A7", N2, N1, N1, N1, N1, N1, N1);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double*******, Layout> a_(a);
//       Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N2 * 1.1), N1, N1,
//       N1,
//                      N1, N1, N1);
//     }
//     time7_noinit = timer.seconds() / R;
//   }
// }

// template <class Layout>
// void run_resizeview_tests8(int N, int R) {
//   const int N1 = N;
//   const int N2 = N1 * N1;
//   const int N4 = N2 * N2;
//   const int N8 = N4 * N4;

//   double time8, time8_noinit, time_raw = 100000.0;
//   {
//     Kokkos::View<double********, Layout> a("A8", N1, N1, N1, N1, N1, N1, N1,
//                                            N1);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double********, Layout> a_(a);
//       Kokkos::resize(a_, int(N1 * 1.1), N1, N1, N1, N1, N1, N1, N1);
//     }
//     time8 = timer.seconds() / R;
//   }
//   {
//     Kokkos::View<double********, Layout> a("A8", N1, N1, N1, N1, N1, N1, N1,
//                                            N1);
//     Kokkos::Timer timer;
//     for (int r = 0; r < R; r++) {
//       Kokkos::View<double********, Layout> a_(a);
//       Kokkos::resize(Kokkos::WithoutInitializing, a_, int(N1 * 1.1), N1, N1,
//       N1,
//                      N1, N1, N1, N1);
//     }
//     time8_noinit = timer.seconds() / R;
//   }
// }

}  // namespace Test
