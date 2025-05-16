#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <format>

#include "diffusion.hpp"
#include "graph.hpp"
#include "log.hpp"
#include "rng.hpp"

using std::fill;
using std::log;
using std::max_element;
using std::vector;

// submodular optimization framework
// A submodular function is a function f: vector<int> -> double
// such that for any S, T, f(S) + f(T) >= f(S ∪ T) + f(S ∩ T)
// A monotone submodular function is a submodular function that is
// non-decreasing

template <typename Fn>
vector<int> greedy_submodular(const Fn &f, int n, int k) {
  vector<char> selected(n, false);
  vector<int> result;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_submodular i: {}", i));
    int best = -1;
    double best_value = -1;
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
    if constexpr (requires { f.checkpoint(); }) {
      f.checkpoint();
    }
  }
  return result;
}

template <typename Fn>
vector<int> greedy_lazy_forward(const Fn &f, int n, int k) {
  vector<char> selected(n, false);
  vector<int> visited(n, 0);
  vector<double> upper_bounds(n, +INFINITY);
  vector<int> result;
  int time = 0;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_lazy_forward i: {}", i));
    auto now = ++time;
    int next_element = -1;
    while (true) {
      int max_ub = max_element(upper_bounds.begin(), upper_bounds.end()) -
                   upper_bounds.begin();
      if (visited[max_ub] < now) {
        auto value = f(vector{max_ub}, result);
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
    if constexpr (requires { f.checkpoint(); }) {
      f.checkpoint();
    }
  }
  return result;
}

struct DiffusionEvaluate {
  const Graph &g;
  DiffusionType type;
  int repeats;
  mutable RNG rng;
  mutable int n_eval = 0;
  mutable vector<size_t> used_evals;
  DiffusionEvaluate(const Graph &g, DiffusionType type, int repeats)
      : g(g), type(type), repeats(repeats), rng() {}
  void seed(seed_type seed) { rng.seed(seed); }
  double operator()(const vector<int> &origin,
                    const vector<int> &prepare = {}) const {
    n_eval++;
    auto solver = DiffusionSolver(g, rng());
    double total = 0;
    for (int i = 0; i < repeats; i++) {
      auto result = solver.run(type, origin, prepare);
      total += result;
    }
    return total / repeats;
  }

  void checkpoint() const { used_evals.push_back(n_eval); }
};

template <typename Greedy> struct GreedyDiffusion {
  int n;
  int k;
  double eps;
  double delta;
  const Greedy &greedy;
  DiffusionEvaluate eval;
  GreedyDiffusion(const Graph &g, DiffusionType diffusion_type, int k,
                  double eps, double delta, const Greedy &greedy)
      : n(g.n), k(k), eps(eps), delta(delta), greedy(greedy),
        eval(g, diffusion_type, 1) {
    std::cout << "n: " << g.n << std::endl;
    std::cout << "eps: " << eps << std::endl;
    std::cout << "delta: " << delta << std::endl;
    std::cout << "repeats: " << g.n * g.n / (eps * eps) * log(g.n * g.n / delta)
              << std::endl;
    eval.repeats = g.n * g.n / (eps * eps) * log(g.n * g.n / delta);
  }

  vector<int> run(seed_type seed) {
    eval.seed(seed);
    return greedy(eval, n, k);
  }

  size_t samples() const { return size_t(eval.n_eval) * eval.repeats; }

  vector<size_t> used_samples() const {
    auto samples = eval.used_evals;
    for (auto &sample : samples) {
      sample *= eval.repeats;
    }
    return samples;
  }
};

// Type deduction rule
template <typename Greedy>
GreedyDiffusion(const Graph &g, DiffusionType diffusion_type, int k, double eps,
                double delta, const Greedy &greedy) -> GreedyDiffusion<Greedy>;
