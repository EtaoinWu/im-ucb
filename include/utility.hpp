#pragma once

#include <vector>
#include <numeric>
// A generic repeat function that takes a function fn
// and repeats it n times, returning a vector of the results
template <typename Fn, typename... Args>
auto repeat(size_t n, Fn &&fn, Args &&...args) {
  std::vector<decltype(fn(args...))> results;
  results.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    results.push_back(fn(args...));
  }
  return results;
}

// A generic repeat function that takes a function fn
// and repeats it n times, returning the average of the results
template <typename Fn, typename... Args>
auto repeat_avg(size_t n, Fn &&fn, Args &&...args) {
  auto results = repeat(n, std::forward<Fn>(fn), std::forward<Args>(args)...);
  return std::accumulate(results.begin(), results.end(), 0.0) / results.size();
}
