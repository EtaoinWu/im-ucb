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

struct VarianceTrackerConfig {
  bool enabled;
  double eta;
  double m;
  double s;
  VarianceTrackerConfig(bool enabled = false,
                        double eta = 2.280,
                        double m = 1.0,
                        double s = 1.418)
      : enabled(enabled), eta(eta), m(m), s(s) {}
};

template <typename Tracker>
struct CBTrackerFactory;

template <>
struct CBTrackerFactory<LILConfidenceBoundTracker> {
  [[nodiscard]] static auto make(int n,
                                 double delta,
                                 const VarianceTrackerConfig&)
      -> LILConfidenceBoundTracker {
    return LILConfidenceBoundTracker(0.03, delta / n, n / 2.0, 0.5, 0.0,
                                     static_cast<double>(n));
  }
};

template <>
struct CBTrackerFactory<PolyStitchingConfidenceBoundTracker> {
  [[nodiscard]] static auto make(int n,
                                 double delta,
                                 const VarianceTrackerConfig& variance_config)
      -> PolyStitchingConfidenceBoundTracker {
    return PolyStitchingConfidenceBoundTracker(
        delta / n, variance_config.eta, variance_config.m, variance_config.s,
        0.0, static_cast<double>(n));
  }
};

template <typename Tracker>
concept CBGreedyTracker =
    ConfidenceBoundTracker<Tracker> &&
    requires(int n, double delta, const VarianceTrackerConfig& config) {
      {
        CBTrackerFactory<Tracker>::make(n, delta, config)
      } -> std::same_as<Tracker>;
    };

template <CBGreedyTracker Tracker>
[[nodiscard]] auto build_trackers(int n,
                                  double delta,
                                  const VarianceTrackerConfig& variance_config)
    -> std::vector<Tracker> {
  std::vector<Tracker> trackers;
  trackers.reserve(n);
  for (int i = 0; i < n; i++) {
    trackers.push_back(
        CBTrackerFactory<Tracker>::make(n, delta, variance_config));
  }
  return trackers;
}

template <CBGreedyTracker Tracker, CBGreedyReward Fn>
[[nodiscard]] auto run_greedy_cb(Fn& f,
                                 int n,
                                 int k,
                                 double eps,
                                 double delta,
                                 const VarianceTrackerConfig& variance_config,
                                 bool lazy_mode,
                                 std::string_view log_prefix)
    -> std::vector<int> {
  auto trackers = build_trackers<Tracker>(n, delta, variance_config);
  UCB<Tracker, Fn> ucb(n, 3.0, eps, f, std::move(trackers), true);
  std::vector<char> selected(n, false);
  std::vector<int> result;
  for (int i = 1; i <= k; i++) {
    my_log(std::format("{} i: {}", log_prefix, i));
    if (!lazy_mode) {
      ucb.reset();
      ucb.enable_all_arms();
      for (int j = 0; j < n; j++) {
        if (selected[j])
          ucb.disable_arm(j);
      }
    }
    auto x = ucb.best_arm();
    selected[x] = true;
    result.push_back(x);
    f.add_fixed(x);
    f.checkpoint();
    if (lazy_mode) {
      ucb.disable_arm(x);
    }
  }
  return result;
}

template <CBGreedyReward Fn,
          CBGreedyTracker Tracker = LILConfidenceBoundTracker>
[[nodiscard]] auto greedy_cb(Fn& f,
                             int n,
                             int k,
                             double eps,
                             double delta,
                             VarianceTrackerConfig variance_config = {})
    -> std::vector<int> {
  return run_greedy_cb<Tracker>(f, n, k, eps, delta, variance_config, false,
                                "greedy_cb");
}

template <CBGreedyReward Fn,
          CBGreedyTracker Tracker = LILConfidenceBoundTracker>
[[nodiscard]] auto greedy_cb_lazy(Fn& f,
                                  int n,
                                  int k,
                                  double eps,
                                  double delta,
                                  VarianceTrackerConfig variance_config = {})
    -> std::vector<int> {
  return run_greedy_cb<Tracker>(f, n, k, eps, delta, variance_config, true,
                                "greedy_cb_lazy");
}

template <typename GreedyCB>
concept GreedyCBSelector = requires(const GreedyCB& cb,
                                    DiffusionReward& reward,
                                    int n,
                                    int k,
                                    double eps,
                                    double delta) {
  requires(
      requires {
        { cb(reward, n, k, eps, delta) } -> std::same_as<std::vector<int>>;
      } ||
      requires {
        {
          cb(reward, n, k, eps, delta, VarianceTrackerConfig{})
        } -> std::same_as<std::vector<int>>;
      });
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
    auto result = [&] {
      if constexpr (requires {
                      {
                        cb_fn(reward, n, k, eps, delta)
                      } -> std::same_as<std::vector<int>>;
                    }) {
        return cb_fn(reward, n, k, eps, delta);
      } else {
        return cb_fn(reward, n, k, eps, delta, VarianceTrackerConfig{});
      }
    }();
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
using im::VarianceTrackerConfig;
