//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_StdAlgorithms.hpp>
#include <TestStdAlgorithmsHelperFunctors.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>

namespace KokkosBenchmark {
namespace stdalgos {
namespace ForEach {

namespace KE = Kokkos::Experimental;

using exespace = Kokkos::DefaultExecutionSpace;

static constexpr auto SIZE = 101513;

template <class ValueType>
static void Algorithms_ForEach(benchmark::State& state) {
  const std::size_t size = state.range(0);
  using view_t           = Kokkos::View<ValueType*>;
  view_t view{"Kokkos::for_each__benchmark", size};

  using view_host_space_t = Kokkos::View<ValueType*, Kokkos::HostSpace>;
  view_host_space_t host_view("for_each_expected", view.extent(0));

  const auto mod_lambda = KOKKOS_LAMBDA(ValueType & i) { ++i; };

  for (auto _ : state) {
    Kokkos::fence();

    Kokkos::Timer timer;
    KE::for_each(exespace(), view, mod_lambda);
    auto time_kokkos = timer.seconds();

    timer.reset();
    std::for_each(KE::begin(host_view), KE::end(host_view), mod_lambda);
    auto time_std = timer.seconds();

    state.counters["Time (Kokkos)"] = benchmark::Counter(time_kokkos);
    state.counters["Time (std)"]    = benchmark::Counter(time_std);
  }
}

BENCHMARK(Algorithms_ForEach<double>)->ArgName("Size")->Arg(SIZE);

}  // namespace ForEach
}  // namespace stdalgos
}  // namespace KokkosBenchmark
