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

#ifndef KOKKOS_STD_ALGORITHMS_MIN_ELEMENT_HPP
#define KOKKOS_STD_ALGORITHMS_MIN_ELEMENT_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_MinMaxMinmaxElement.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class IteratorType,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last) {
  return Impl::min_or_max_element_exespace_impl<MinFirstLoc>(
      "Kokkos::min_element_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last) {
  return Impl::min_or_max_element_exespace_impl<MinFirstLoc>(label, ex, first,
                                                             last);
}

template <class ExecutionSpace, class IteratorType, class ComparatorType,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last, ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::min_or_max_element_exespace_impl<MinFirstLocCustomComparator>(
      "Kokkos::min_element_iterator_api_default", ex, first, last,
      std::move(comp));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last, ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::min_or_max_element_exespace_impl<MinFirstLocCustomComparator>(
      label, ex, first, last, std::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::min_or_max_element_exespace_impl<MinFirstLoc>(
      "Kokkos::min_element_view_api_default", ex, begin(v), end(v));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::min_or_max_element_exespace_impl<MinFirstLocCustomComparator>(
      "Kokkos::min_element_view_api_default", ex, begin(v), end(v),
      std::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::min_or_max_element_exespace_impl<MinFirstLoc>(label, ex,
                                                             begin(v), end(v));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::min_or_max_element_exespace_impl<MinFirstLocCustomComparator>(
      label, ex, begin(v), end(v), std::move(comp));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <
    class TeamHandleType, class IteratorType,
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto min_element(const TeamHandleType& teamHandle,
                                 IteratorType first, IteratorType last) {
  return Impl::min_or_max_element_team_impl<MinFirstLoc>(teamHandle, first,
                                                         last);
}

template <
    class TeamHandleType, class DataType, class... Properties,
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto min_element(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::min_or_max_element_team_impl<MinFirstLoc>(teamHandle, begin(v),
                                                         end(v));
}

// for OpenMPTarget we cannot have a custom comparator
#if not defined KOKKOS_ENABLE_OPENMPTARGET
template <
    class TeamHandleType, class IteratorType, class ComparatorType,
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto min_element(const TeamHandleType& teamHandle,
                                 IteratorType first, IteratorType last,
                                 ComparatorType comp) {
  return Impl::min_or_max_element_team_impl<MinFirstLocCustomComparator>(
      teamHandle, first, last, std::move(comp));
}

template <
    class TeamHandleType, class DataType, class ComparatorType,
    class... Properties,
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto min_element(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v, ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  return Impl::min_or_max_element_team_impl<MinFirstLocCustomComparator>(
      teamHandle, begin(v), end(v), std::move(comp));
}
#endif

}  // namespace Experimental
}  // namespace Kokkos

#endif
