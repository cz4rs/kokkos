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

#include <gtest/gtest.h>
#include <impl/Kokkos_CheckedIntegerOps.hpp>
#include <limits>

namespace Test {

TEST(TEST_CATEGORY, test_multiply) {
  {
    auto result      = 1u;
    auto is_overflow = Kokkos::Impl::multiply_overflow(1u, 2u, result);
    EXPECT_EQ(result, 2u);
    EXPECT_FALSE(is_overflow);
  }
  {
    auto result      = 1u;
    auto is_overflow = Kokkos::Impl::multiply_overflow(
        std::numeric_limits<unsigned>::max(), 2u, result);
    EXPECT_EQ(result, 1u);
    EXPECT_TRUE(is_overflow);
  }
}

}  // namespace Test
