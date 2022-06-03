#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include <PerfTest_ViewCopy.hpp>

static void ViewDeepCopy_RightLeft_Rank8(benchmark::State& state) {
  // Perform setup here
  printf("DeepCopy Performance for LayoutRight to LayoutLeft:\n");

  for (auto _ : state) {
    // This code gets timed
    Test::run_deepcopyview_tests8<Kokkos::LayoutRight, Kokkos::LayoutLeft>(10,
                                                                           1);
  }
}

// Register the function as a benchmark
BENCHMARK(ViewDeepCopy_RightLeft_Rank8);

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
