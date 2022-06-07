#include <benchmark/benchmark.h>

#include <cmath>

#include <Kokkos_Core.hpp>

#include <PerfTest_ViewCopy.hpp>

// replicates core/perf_test/PerfTest_ViewCopy_d8.cpp

void report_results(benchmark::State& state, double time) {
  state.SetIterationTime(time);

  const auto N8        = std::pow(state.range(0), 8);
  const auto size      = N8 * 8 / 1024 / 1024;
  state.counters["MB"] = benchmark::Counter(size, benchmark::Counter::kDefaults,
                                            benchmark::Counter::OneK::kIs1024);
  state.counters["GB/s"] =
      benchmark::Counter(2 * size / 1024 / time, benchmark::Counter::kDefaults,
                         benchmark::Counter::OneK::kIs1024);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank8(benchmark::State& state) {
  const auto N = state.range(0);
  const auto R = state.range(1);

  const int N1 = N;
  Kokkos::View<double********, LayoutA> a("A8", N1, N1, N1, N1, N1, N1, N1, N1);
  Kokkos::View<double********, LayoutB> b("B8", N1, N1, N1, N1, N1, N1, N1, N1);

  for (auto _ : state) {
    const auto elapsed_seconds = Test::deepcopy_view(a, b, R);

    report_results(state, elapsed_seconds);
  }
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank8_Raw(benchmark::State& state) {
  const auto N = state.range(0);
  const auto R = state.range(1);

  const int N1 = N;
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;
  Kokkos::View<double*, LayoutA> a("A1", N8);
  Kokkos::View<double*, LayoutB> b("B1", N8);
  double* const a_ptr       = a.data();
  const double* const b_ptr = b.data();

  for (auto _ : state) {
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = b_ptr[i]; });
    }
    Kokkos::fence();

    report_results(state, timer.seconds());
  }
}

BENCHMARK(ViewDeepCopy_Rank8<Kokkos::LayoutRight, Kokkos::LayoutLeft>)
    ->ArgNames({"N", "R"})
    ->Args({10, 1})
    ->UseManualTime();

#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
BENCHMARK(ViewDeepCopy_Rank8_Raw<Kokkos::LayoutRight, Kokkos::LayoutLeft>)
    ->ArgNames({"N", "R"})
    ->Args({10, 1})
    ->UseManualTime();
#endif

std::string custom_context() {
  std::ostringstream msg;
  Kokkos::print_configuration(msg);
  return msg.str();
}

// Run benchmarks
// BENCHMARK_MAIN() + Kokkos init / finalize
int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  Kokkos::initialize(argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kSecond);
  benchmark::AddCustomContext("Kokkos configuration", custom_context());

  benchmark::RunSpecifiedBenchmarks();
  // REMOVE_ME: Run the vanilla test for comparison
  Test::run_deepcopyview_tests8<Kokkos::LayoutRight, Kokkos::LayoutLeft>(10, 1);

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
