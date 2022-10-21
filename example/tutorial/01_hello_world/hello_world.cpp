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
#include <list>

using Device = Kokkos::DefaultHostExecutionSpace;

void driver(){
    uint32_t n = 200000000;
    Kokkos::View<uint32_t*, Device> a("test", n);
    std::list<Kokkos::View<uint32_t*, Device>> sources;
    uint32_t t_iter = 10;
    Kokkos::fence();
    Kokkos::Timer t;
    double begin = t.seconds();
    uint32_t segments = 64;
    uint32_t width = n / segments;
    for(uint32_t i = 0; i < t_iter; i++){
        for(uint32_t j = 0; j < segments; j++){
            auto source = Kokkos::subview(a, std::make_pair(width*j, width*(j+1)));
            Kokkos::View<uint32_t*, Device> dest(Kokkos::ViewAllocateWithoutInitializing("dest"), width);
            Kokkos::deep_copy(dest, source);
            sources.push_back(dest);
        }
        Kokkos::fence();
    }
    double end = t.seconds();
    double d = end - begin;
    double avg = d / static_cast<double>(t_iter);
    double bandwidth = static_cast<double>(2*sizeof(uint32_t)*n) / avg;
    printf("uint32_t size: %li\n", sizeof(uint32_t));
    printf("Total time: %.2fs; Avg Time: %.2fs; Bandwidth: %.2fMB/s\n", d, avg, bandwidth / 1000000);
}

int main() {
    Kokkos::initialize();
    {
        driver();
    }
    Kokkos::finalize();
    return 0;
}
