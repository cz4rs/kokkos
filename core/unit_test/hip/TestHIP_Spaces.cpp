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
#include <TestHIP_Category.hpp>

namespace Test {

__global__ void test_abort() { Kokkos::abort("test_abort"); }

__global__ void test_hip_spaces_int_value(int *ptr) {
  if (*ptr == 42) {
    *ptr = 2 * 42;
  }
}

TEST(hip, space_access) {
  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                Kokkos::HostSpace>::assignable,
                "");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                      Kokkos::HIPHostPinnedSpace>::assignable,
      "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                 Kokkos::HIPSpace>::assignable,
                "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                 Kokkos::HIPSpace>::accessible,
                "");

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                       Kokkos::HIPManagedSpace>::assignable,
      "");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                      Kokkos::HIPManagedSpace>::accessible,
      "");

  //--------------------------------------

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                                Kokkos::HIPSpace>::assignable,
                "");

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                       Kokkos::HIPHostPinnedSpace>::assignable,
      "");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                      Kokkos::HIPHostPinnedSpace>::accessible,
      "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                                 Kokkos::HostSpace>::assignable,
                "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                                 Kokkos::HostSpace>::accessible,
                "");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                      Kokkos::HIPManagedSpace>::assignable,
      "");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                      Kokkos::HIPManagedSpace>::accessible,
      "");

  //--------------------------------------

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                      Kokkos::HIPHostPinnedSpace>::assignable,
      "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                                 Kokkos::HostSpace>::assignable,
                "");

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                                Kokkos::HostSpace>::accessible,
                "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                                 Kokkos::HIPSpace>::assignable,
                "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                                 Kokkos::HIPSpace>::accessible,
                "");

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                       Kokkos::HIPManagedSpace>::assignable,
      "");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                      Kokkos::HIPManagedSpace>::accessible,
      "");

  //--------------------------------------

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                      Kokkos::HIPManagedSpace>::assignable,
      "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                                 Kokkos::HostSpace>::assignable,
                "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                                 Kokkos::HostSpace>::accessible,
                "");

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                                 Kokkos::HIPSpace>::assignable,
                "");

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                                Kokkos::HIPSpace>::accessible,
                "");

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                       Kokkos::HIPHostPinnedSpace>::assignable,
      "");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                      Kokkos::HIPHostPinnedSpace>::accessible,
      "");

  //--------------------------------------

  static_assert(
      !Kokkos::SpaceAccessibility<Kokkos::HIP, Kokkos::HostSpace>::accessible,
      "");

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HIP, Kokkos::HIPSpace>::accessible,
      "");

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HIP,
                                 Kokkos::HIPHostPinnedSpace>::accessible,
      "");

  static_assert(Kokkos::SpaceAccessibility<Kokkos::HIP,
                                           Kokkos::HIPManagedSpace>::accessible,
                "");

  static_assert(!Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                            Kokkos::HIPSpace>::accessible,
                "");

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 Kokkos::HIPHostPinnedSpace>::accessible,
      "");

  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                           Kokkos::HIPManagedSpace>::accessible,
                "");

  static_assert(std::is_same<Kokkos::Impl::HostMirror<Kokkos::HIPSpace>::Space,
                             Kokkos::HostSpace>::value,
                "");

  static_assert(
      std::is_same<Kokkos::Impl::HostMirror<Kokkos::HIPHostPinnedSpace>::Space,
                   Kokkos::HIPHostPinnedSpace>::value,
      "");

  static_assert(
      std::is_same<Kokkos::Impl::HostMirror<Kokkos::HIPManagedSpace>::Space,
                   Kokkos::Device<Kokkos::HostSpace::execution_space,
                                  Kokkos::HIPManagedSpace>>::value,
      "");

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::Impl::HostMirror<Kokkos::HIP>::Space,
                                 Kokkos::HostSpace>::accessible,
      "");

  static_assert(Kokkos::SpaceAccessibility<
                    Kokkos::Impl::HostMirror<Kokkos::HIPSpace>::Space,
                    Kokkos::HostSpace>::accessible,
                "");

  static_assert(Kokkos::SpaceAccessibility<
                    Kokkos::Impl::HostMirror<Kokkos::HIPHostPinnedSpace>::Space,
                    Kokkos::HostSpace>::accessible,
                "");

  static_assert(Kokkos::SpaceAccessibility<
                    Kokkos::Impl::HostMirror<Kokkos::HIPManagedSpace>::Space,
                    Kokkos::HostSpace>::accessible,
                "");
}

template <class MemSpace, class ExecSpace>
struct TestViewHIPAccessible {
  enum { N = 1000 };

  using V = Kokkos::View<double *, MemSpace>;

  V m_base;

  struct TagInit {};
  struct TagTest {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagInit &, const int i) const { m_base[i] = i + 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagTest &, const int i, long &error_count) const {
    if (m_base[i] != i + 1) ++error_count;
  }

  TestViewHIPAccessible() : m_base("base", N) {}

  static void run() {
    TestViewHIPAccessible self;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<typename MemSpace::execution_space, TagInit>(0, N),
        self);
    typename MemSpace::execution_space().fence();

    // Next access is a different execution space, must complete prior kernel.
    long error_count = -1;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpace, TagTest>(0, N), self,
                            error_count);
    EXPECT_EQ(error_count, 0);
  }
};

TEST(hip, impl_view_accessible) {
  TestViewHIPAccessible<Kokkos::HIPSpace, Kokkos::HIP>::run();

  TestViewHIPAccessible<Kokkos::HIPHostPinnedSpace, Kokkos::HIP>::run();
  TestViewHIPAccessible<Kokkos::HIPHostPinnedSpace,
                        Kokkos::HostSpace::execution_space>::run();

  TestViewHIPAccessible<Kokkos::HIPManagedSpace,
                        Kokkos::HostSpace::execution_space>::run();
  TestViewHIPAccessible<Kokkos::HIPManagedSpace, Kokkos::HIP>::run();
}

}  // namespace Test
