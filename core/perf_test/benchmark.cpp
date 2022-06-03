#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

static void BM_SomeFunction(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    // SomeFunction();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_SomeFunction);

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
