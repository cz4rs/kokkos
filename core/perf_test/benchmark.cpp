#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include <PerfTest_ViewCopy.hpp>

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank8(benchmark::State& state) {
  // Perform setup here
  // ...
  for (auto _ : state) {
    // This code gets timed
    Test::run_deepcopyview_tests8<LayoutA, LayoutB>(10, 1);
  }
}

// Register the function as a benchmark
BENCHMARK(ViewDeepCopy_Rank8<Kokkos::LayoutRight, Kokkos::LayoutLeft>);

// Run benchmarks
// BENCHMARK_MAIN() + Kokkos init / finalize
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  Kokkos::initialize(argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  ::benchmark::RunSpecifiedBenchmarks();

  ::benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
