#pragma once

#include <concepts>
#include <format>
#include <vector>

#include "diffusion.hpp"
#include "graph.hpp"
#include "log.hpp"
#include "rng.hpp"
#include "ucb.hpp"

namespace im {

// A wrapper of DiffusionSolver to be used as the reward function for
// confidence-bound-based algorithms. It exposes a single-arm reward interface.
struct DiffusionReward {
  DiffusionSolver& solver;
  DiffusionType type;
  std::vector<int> fixed_vertices;
  size_t samples;
  std::vector<size_t> used_samples;
  DiffusionReward(DiffusionSolver& solver,
                  DiffusionType type,
                  std::vector<int> fixed_vertices = {})
      : solver(solver),
        type(type),
        fixed_vertices(fixed_vertices),
        samples(0),
        used_samples() {}

  [[nodiscard]] auto operator()(int i) -> double {
    samples++;
    return solver.run(type, i, fixed_vertices);
  }

  auto checkpoint() -> void { used_samples.push_back(samples); }

  auto add_fixed(int i) -> void { fixed_vertices.push_back(i); }
};

template <typename Fn>
concept CBGreedyReward = requires(Fn& fn, int i) {
  { fn(i) } -> std::convertible_to<double>;
  { fn.add_fixed(i) } -> std::same_as<void>;
  { fn.checkpoint() } -> std::same_as<void>;
};

static_assert(CBGreedyReward<DiffusionReward>,
              "DiffusionReward does not satisfy CBGreedyReward");

template <CBGreedyReward Fn>
[[nodiscard]] auto greedy_cb(Fn& f, int n, int k, double eps, double delta)
    -> std::vector<int> {
  using Tracker = LILConfidenceBoundTracker;
  std::vector<Tracker> trackers;
  trackers.reserve(n);
  for (int i = 0; i < n; i++) {
    trackers.emplace_back(0.03, delta / n, n / 2.0, 0.5, 0.0,
                          static_cast<double>(n));
  }
  UCB<Tracker, Fn> ucb(n, 3.0, eps, f, std::move(trackers), true);
  std::vector<char> selected(n, false);
  std::vector<int> result;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_cb i: {}", i));
    ucb.reset();
    ucb.enable_all_arms();
    for (int j = 0; j < n; j++) {
      if (selected[j])
        ucb.disable_arm(j);
    }
    auto x = ucb.best_arm();
    selected[x] = true;
    result.push_back(x);
    f.add_fixed(x);
    f.checkpoint();
  }
  return result;
}

template <CBGreedyReward Fn>
[[nodiscard]] auto greedy_cb_lazy(Fn& f, int n, int k, double eps, double delta)
    -> std::vector<int> {
  using Tracker = LILConfidenceBoundTracker;
  std::vector<Tracker> trackers;
  trackers.reserve(n);
  for (int i = 0; i < n; i++) {
    trackers.emplace_back(0.03, delta / n, n / 2.0, 0.5, 0.0,
                          static_cast<double>(n));
  }
  UCB<Tracker, Fn> ucb(n, 3.0, eps, f, std::move(trackers), true);
  std::vector<char> selected(n, false);
  std::vector<int> result;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("greedy_cb_lazy i: {}", i));
    auto x = ucb.best_arm();
    selected[x] = true;
    result.push_back(x);
    f.add_fixed(x);
    f.checkpoint();
    ucb.disable_arm(x);
  }
  return result;
}

template <typename GreedyCB>
concept GreedyCBSelector = requires(const GreedyCB& cb,
                                    DiffusionReward& reward,
                                    int n,
                                    int k,
                                    double eps,
                                    double delta) {
  { cb(reward, n, k, eps, delta) } -> std::same_as<std::vector<int>>;
};

template <GreedyCBSelector GreedyCB>
struct GreedyCBDiffusion {
  int n;
  int k;
  DiffusionType type;
  double eps;
  double delta;
  const GreedyCB& cb_fn;
  DiffusionSolver solver;
  size_t total_samples;
  std::vector<size_t> used_samples_;
  GreedyCBDiffusion(const Graph& g,
                    DiffusionType diffusion_type,
                    int k,
                    double eps,
                    double delta,
                    const GreedyCB& cb_fn)
      : n(g.n),
        k(k),
        type(diffusion_type),
        eps(eps),
        delta(delta),
        cb_fn(cb_fn),
        solver(g, 0),
        total_samples(0),
        used_samples_() {}

  [[nodiscard]] auto run(seed_type seed) -> std::vector<int> {
    solver.seed(seed);
    auto reward = DiffusionReward(solver, type);
    auto result = cb_fn(reward, n, k, eps, delta);
    total_samples += reward.samples;
    used_samples_.insert(used_samples_.end(), reward.used_samples.begin(),
                         reward.used_samples.end());
    return result;
  }

  [[nodiscard]] auto samples() const -> size_t { return total_samples; }
  [[nodiscard]] auto used_samples() const -> std::vector<size_t> {
    return used_samples_;
  }
};

template <typename GreedyCB>
GreedyCBDiffusion(const Graph& g,
                  DiffusionType diffusion_type,
                  int k,
                  double eps,
                  double delta,
                  const GreedyCB& cb_fn) -> GreedyCBDiffusion<GreedyCB>;

}  // namespace im

using im::DiffusionReward;
using im::greedy_cb;
using im::greedy_cb_lazy;
using im::GreedyCBDiffusion;
