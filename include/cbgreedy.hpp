#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <vector>

#include "diffusion.hpp"
#include "graph.hpp"
#include "log.hpp"
#include "rng.hpp"
#include "ucb.hpp"

using std::vector;

struct DiffusionReward {
  DiffusionSolver &solver;
  DiffusionType type;
  vector<int> fixed_vertices;
  mutable size_t samples;
  vector<size_t> used_samples;
  DiffusionReward(DiffusionSolver &solver, DiffusionType type,
                  vector<int> fixed_vertices = {})
      : solver(solver), type(type), fixed_vertices(fixed_vertices), samples(0),
        used_samples() {}

  double operator()(int i) const {
    samples++;
    return solver.run(type, vector{i}, fixed_vertices);
  }

  void checkpoint() { used_samples.push_back(samples); }

  void add_fixed(int i) { fixed_vertices.push_back(i); }
};

template <typename Fn>
vector<int> greedy_cb(Fn &f, int n, int k, double eps, double delta) {
  auto lil = LILConfidence(0.03, delta / n, n / 2.0);
  UCB<LILConfidence, Fn> ucb(n, 3.0, 0.5, eps, lil, f, {}, true);
  vector<char> selected(n, false);
  vector<int> result;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_cb i: {}", i));
    ucb.reset();
    for (int j = 0; j < n; j++) {
      if (selected[j])
        ucb.upper_bounds[j] = -1;
    }
    auto x = ucb.best_arm();
    selected[x] = true;
    result.push_back(x);
    f.add_fixed(x);
    f.checkpoint();
  }
  return result;
}

template <typename Fn>
vector<int> greedy_cb_lazy(Fn &f, int n, int k, double eps, double delta) {
  auto lil = LILConfidence(0.03, delta / n, n / 2.0);
  UCB<LILConfidence, Fn> ucb(n, 3.0, 0.5, eps, lil, f, {}, true);
  vector<char> selected(n, false);
  vector<int> result;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_cb_lazy i: {}", i));
    auto x = ucb.best_arm();
    selected[x] = true;
    result.push_back(x);
    f.add_fixed(x);
    f.checkpoint();
    ucb.upper_bounds[x] = -1;
  }
  return result;
}

template <typename GreedyCB> struct GreedyCBDiffusion {
  int n;
  int k;
  DiffusionType type;
  double eps;
  double delta;
  const GreedyCB &cb_fn;
  DiffusionSolver solver;
  size_t total_samples;
  vector<size_t> used_samples_;
  GreedyCBDiffusion(const Graph &g, DiffusionType diffusion_type, int k,
                    double eps, double delta, const GreedyCB &cb_fn)
      : n(g.n), k(k), type(diffusion_type), eps(eps), delta(delta),
        cb_fn(cb_fn), solver(g, 0), total_samples(0), used_samples_() {}

  vector<int> run(seed_type seed) {
    solver.seed(seed);
    auto reward = DiffusionReward(solver, type);
    auto result = cb_fn(reward, n, k, eps, delta);
    total_samples += reward.samples;
    used_samples_.insert(used_samples_.end(), reward.used_samples.begin(),
                         reward.used_samples.end());
    return result;
  }

  size_t samples() const { return total_samples; }
  vector<size_t> used_samples() const { return used_samples_; }
};

template <typename GreedyCB>
GreedyCBDiffusion(const Graph &g, DiffusionType diffusion_type, int k,
                  double eps, double delta, const GreedyCB &cb_fn)
    -> GreedyCBDiffusion<GreedyCB>;
