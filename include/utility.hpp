#pragma once

#include <concepts>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

// A generic repeat function that takes a function fn
// and repeats it n times, returning a vector of the results
template <typename Fn, typename... Args>
  requires std::invocable<Fn &, Args...>
auto repeat(size_t n, Fn &&fn, Args &&...args) {
  using result_t = std::invoke_result_t<Fn &, Args...>;
  std::vector<result_t> results;
  results.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    results.push_back(std::invoke(fn, args...));
  }
  return results;
}

// A generic repeat function that takes a function fn
// and repeats it n times, returning the average of the results
template <typename Fn, typename... Args>
  requires std::invocable<Fn &, Args...>
auto repeat_avg(size_t n, Fn &&fn, Args &&...args) {
  auto results = repeat(n, std::forward<Fn>(fn), std::forward<Args>(args)...);
  return std::accumulate(results.begin(), results.end(), 0.0) / results.size();
}
