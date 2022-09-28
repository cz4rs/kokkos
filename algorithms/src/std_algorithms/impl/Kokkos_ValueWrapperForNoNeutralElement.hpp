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

#ifndef KOKKOS_STD_ALGORITHMS_VALUE_WRAPPER_FOR_NO_NEUTRAL_ELEMENT_HPP
#define KOKKOS_STD_ALGORITHMS_VALUE_WRAPPER_FOR_NO_NEUTRAL_ELEMENT_HPP

namespace Kokkos {
namespace Experimental {
namespace Impl {

//
// scalar wrapper used for reductions and scans
// when we don't have neutral element
//
template <class Scalar>
struct ValueWrapperForNoNeutralElement {
  Scalar val;
  bool is_initial = true;

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement() = default;

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement(const ValueWrapperForNoNeutralElement&) =
      default;

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement(ValueWrapperForNoNeutralElement&&) = default;

  KOKKOS_FUNCTION
  ~ValueWrapperForNoNeutralElement() = default;

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement& operator=(
      const ValueWrapperForNoNeutralElement& rhs) {
    val        = rhs.val;
    is_initial = rhs.is_initial;
    return *this;
  }

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement& operator=(
      ValueWrapperForNoNeutralElement&& rhs) {
    val        = std::move(rhs.val);
    is_initial = rhs.is_initial;
    return *this;
  }

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement(int) {}

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement(Scalar v, bool is_init)
      : val{v}, is_initial{is_init} {}

  KOKKOS_FUNCTION
  ValueWrapperForNoNeutralElement& operator+=(
      const ValueWrapperForNoNeutralElement& rhs) {
    val += rhs.val;
    return *this;
  }

  KOKKOS_FUNCTION
  friend ValueWrapperForNoNeutralElement operator+(
      ValueWrapperForNoNeutralElement lhs,
      const ValueWrapperForNoNeutralElement& rhs) {
    lhs += rhs;
    return lhs;
  }
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
