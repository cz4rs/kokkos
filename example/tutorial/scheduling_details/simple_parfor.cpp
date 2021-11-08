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

//
// First Kokkos::View (multidimensional array) example:
//   1. Start up Kokkos
//   2. Allocate a Kokkos::View
//   3. Execute a parallel_for over that View's data
//   4. Shut down Kokkos
//
// Compare this example to 03_simple_view_lambda, which uses C++11
// lambdas to define the loop bodies of the parallel_for and
// parallel_reduce.
//

#include <Kokkos_Core.hpp>
#include <cstdio>

using view_type = Kokkos::View<double * [3]>;

struct SimpleFunctor {
  view_type a;

  SimpleFunctor(view_type a_) : a(a_) {}

  // Fill the View with some data.  The parallel_for loop will iterate
  // over the View's first dimension N.
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    a(i, 0) = 1.0 * i;
    a(i, 1) = 1.0 * i * i;
    a(i, 2) = 1.0 * i * i * i;
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  // {
  //   const int N = 1003;
  //   view_type a("scheduling-details-static", N);

  //   Kokkos::parallel_for(N, SimpleFunctor(a));
  // }

  {
    const int N = 33;
    view_type a("scheduling-details-dynamic", N);
    using policy_t = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                                         Kokkos::Schedule<Kokkos::Dynamic> >;

    Kokkos::parallel_for(policy_t(0, N), SimpleFunctor(a));
  }

  Kokkos::finalize();
}
