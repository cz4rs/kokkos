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

namespace Test {

void report_results_fill(benchmark::State& state, double time);

template <class ViewType>
void fill_view(ViewType& a, typename ViewType::const_value_type& val,
               benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::deep_copy(a, val);
    Kokkos::fence();
    report_results_fill(state, timer.seconds());
  }
}

template <class Layout>
static void ViewFill_Rank1(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;

  Kokkos::View<double*, Layout> a("A1", N8);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Rank2(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;

  Kokkos::View<double**, Layout> a("A2", N4, N4);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Rank3(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;
  const int N3 = N2 * N1;

  Kokkos::View<double***, Layout> a("A3", N3, N3, N2);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Rank4(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double****, Layout> a("A4", N2, N2, N2, N2);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Rank5(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double*****, Layout> a("A5", N2, N2, N1, N1, N2);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Rank6(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double******, Layout> a("A6", N2, N1, N1, N1, N1, N2);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Rank7(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double*******, Layout> a("A7", N2, N1, N1, N1, N1, N1, N1);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Rank8(benchmark::State& state) {
  const int N1 = state.range(0);

  Kokkos::View<double********, Layout> a("A8", N1, N1, N1, N1, N1, N1, N1, N1);
  fill_view(a, 1.1, state);
}

template <class Layout>
static void ViewFill_Raw(benchmark::State& state) {
  const int N8 = std::pow(state.range(0), 8);

  Kokkos::View<double*, Layout> a("A1", N8);
  double* a_ptr = a.data();

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = 1.1; });
    Kokkos::fence();

    report_results_fill(state, timer.seconds());
  }
}

}  // namespace Test
