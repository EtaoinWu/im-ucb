#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <format>
#include <iostream>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

#include "diffusion.hpp"
#include "graph.hpp"
#include "log.hpp"
#include "rng.hpp"

namespace im {

template <typename Fn>
concept Checkpointable = requires(Fn& fn) {
  { fn.checkpoint() } -> std::same_as<void>;
};

template <typename Fn>
concept SubmodularFn = requires(const Fn& fn, const std::vector<int>& set) {
  // submodularity: f(S) + f(T) >= f(S ∪ T) + f(S ∩ T)
  // monotonicity: f(S) <= f(T) for S ⊆ T
  { fn(set) } -> std::convertible_to<double>;
};

template <typename Fn>
concept SubmodularIncrementFn = requires(const Fn& fn,
                                         const std::vector<int>& delta,
                                         const std::vector<int>& base) {
  // calculate incremental value: f(base ∪ delta) - f(base)
  // submodularity: f(A, S) >= f(B, S) for A ⊆ B
  // monotonicity: f(A, S) >= 0
  { fn(delta, base) } -> std::convertible_to<double>;
};

template <SubmodularFn Fn>
[[nodiscard]] auto greedy_submodular(const Fn& f, int n, int k)
    -> std::vector<int> {
  // the standard greedy algorithm for submodular optimization
  std::vector<char> selected(n, false);
  std::vector<int> result;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_submodular i: {}", i));
    int best = -1;
    double best_value = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < n; j++) {
      if (selected[j])
        continue;
      auto new_result = result;
      new_result.push_back(j);
      auto value = f(new_result);
      if (value > best_value) {
        best_value = value;
        best = j;
      }
    }
    selected[best] = true;
    result.push_back(best);
    if constexpr (Checkpointable<const Fn>) {
      f.checkpoint();
    }
  }
  return result;
}

template <SubmodularIncrementFn Fn>
[[nodiscard]] auto greedy_lazy_forward(const Fn& f, int n, int k)
    -> std::vector<int> {
  // the CELF algorithm
  std::vector<char> selected(n, false);
  std::vector<int> visited(n, 0);
  std::vector<double> upper_bounds(n, std::numeric_limits<double>::infinity());
  std::vector<int> result;
  int time = 0;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_lazy_forward i: {}", i));
    auto now = ++time;
    int next_element = -1;
    while (true) {
      int max_ub =
          std::ranges::max_element(upper_bounds) - upper_bounds.begin();
      if (visited[max_ub] < now) {
        auto value = f(std::vector{max_ub}, result);
        upper_bounds[max_ub] = value;
        visited[max_ub] = now;
      } else {
        next_element = max_ub;
        break;
      }
    }
    if (next_element == -1) {
      break;
    }
    selected[next_element] = true;
    result.push_back(next_element);
    upper_bounds[next_element] = -1;
    if constexpr (Checkpointable<const Fn>) {
      f.checkpoint();
    }
  }
  return result;
}

// A wrapper of DiffusionSolver to be used as the reward function for
// generic submodular optimization algorithms. It exposes a set-function
// interface.
struct DiffusionSubmodular {
  const Graph& g;
  DiffusionType type;
  int repeats;
  mutable RNG rng;
  mutable int n_eval = 0;
  mutable std::vector<size_t> used_evals;
  DiffusionSubmodular(const Graph& g, DiffusionType type, int repeats)
      : g(g), type(type), repeats(repeats), rng() {}
  auto seed(seed_type seed) -> void { rng.seed(seed); }

  [[nodiscard]] auto operator()(std::span<const int> origin,
                                std::span<const int> prepare = {}) const
      -> double {
    n_eval++;
    auto solver = DiffusionSolver(g, rng());
    double total = 0;
    for (int i = 0; i < repeats; i++) {
      auto result = solver.run(type, origin, prepare);
      total += result;
    }
    return total / repeats;
  }

  [[nodiscard]] auto operator()(const std::vector<int>& origin,
                                const std::vector<int>& prepare = {}) const
      -> double {
    return (*this)(std::span<const int>(origin), std::span<const int>(prepare));
  }

  auto checkpoint() const -> void { used_evals.push_back(n_eval); }
};

static_assert(SubmodularFn<DiffusionSubmodular>);
static_assert(SubmodularIncrementFn<DiffusionSubmodular>);

template <typename Algo, typename Fn>
concept SubmodularOptAlgo =
    SubmodularFn<Fn> && requires(const Algo& algo, Fn& eval, int n, int k) {
      { algo(eval, n, k) } -> std::same_as<std::vector<int>>;
    };

template <typename Algo>
  requires SubmodularOptAlgo<Algo, DiffusionSubmodular>
struct DiffusionAlgoRun {
  int n;
  int k;
  double eps;
  double delta;
  const Algo& alg;
  DiffusionSubmodular eval;
  DiffusionAlgoRun(const Graph& g,
                   DiffusionType diffusion_type,
                   int k,
                   double eps,
                   double delta,
                   const Algo& alg)
      : n(g.n),
        k(k),
        eps(eps),
        delta(delta),
        alg(alg),
        eval(g, diffusion_type, 1) {
    std::cout << "n: " << g.n << std::endl;
    std::cout << "eps: " << eps << std::endl;
    std::cout << "delta: " << delta << std::endl;
    std::cout << "repeats: "
              << g.n * g.n / (eps * eps) * std::log(g.n * g.n / delta)
              << std::endl;
    eval.repeats = g.n * g.n / (eps * eps) * std::log(g.n * g.n / delta);
  }

  [[nodiscard]] auto run(seed_type seed) -> std::vector<int> {
    eval.seed(seed);
    return alg(eval, n, k);
  }

  [[nodiscard]] auto samples() const -> size_t {
    return static_cast<size_t>(eval.n_eval) * eval.repeats;
  }

  [[nodiscard]] auto used_samples() const -> std::vector<size_t> {
    auto samples = eval.used_evals;
    for (auto& sample : samples) {
      sample *= eval.repeats;
    }
    return samples;
  }
};

// Type deduction rule
template <typename Algo>
DiffusionAlgoRun(const Graph& g,
                 DiffusionType diffusion_type,
                 int k,
                 double eps,
                 double delta,
                 const Algo& alg) -> DiffusionAlgoRun<Algo>;

}  // namespace im

using im::DiffusionAlgoRun;
using im::DiffusionSubmodular;
using im::greedy_lazy_forward;
using im::greedy_submodular;
