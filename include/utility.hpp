#pragma once

#include <cassert>
#include <concepts>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

template <typename Fn, typename... Args>
concept RepeatInvocable = std::invocable<Fn &, Args...>;

template <typename Fn, typename... Args>
concept RepeatAveragable =
  RepeatInvocable<Fn, Args...> &&
  std::convertible_to<std::invoke_result_t<Fn &, Args...>, double>;

// A generic repeat function that takes a function fn
// and repeats it n times, returning a vector of the results
template <typename Fn, typename... Args>
  requires RepeatInvocable<Fn, Args...>
[[nodiscard]] auto repeat(size_t n, Fn &&fn, Args &&...args)
    -> std::vector<std::invoke_result_t<Fn &, Args...>> {
  using result_t = std::invoke_result_t<Fn &, Args...>;
  std::vector<result_t> results;
  results.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    results.push_back(std::invoke(std::forward<Fn>(fn),
                                  std::forward<Args>(args)...));
  }
  return results;
}

// A generic repeat function that takes a function fn
// and repeats it n times, returning the average of the results
template <typename Fn, typename... Args>
  requires RepeatAveragable<Fn, Args...>
[[nodiscard]] auto repeat_avg(size_t n, Fn &&fn, Args &&...args) -> double {
  assert(n > 0);
  auto results = repeat(n, std::forward<Fn>(fn), std::forward<Args>(args)...);
  return std::accumulate(results.begin(), results.end(), 0.0) / results.size();
}
