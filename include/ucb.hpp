#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <format>
#include <vector>

#include "confidence_bound.hpp"
#include "log.hpp"

namespace im {

template <typename Tracker>
concept ConfidenceBoundTracker = requires(Tracker& tracker, double sample) {
  { tracker.add_sample(sample) } -> std::same_as<void>;
  { tracker.mean() } -> std::convertible_to<double>;
  { tracker.ucb() } -> std::convertible_to<double>;
  { tracker.lcb() } -> std::convertible_to<double>;
  { tracker.num_pulls() } -> std::convertible_to<size_t>;
  { tracker.reset_ucb() } -> std::same_as<void>;
};

template <typename Reward>
concept ArmReward = requires(Reward& reward, int arm) {
  { reward(arm) } -> std::convertible_to<double>;
};

static_assert(ConfidenceBoundTracker<LILConfidenceBoundTracker>);
static_assert(ConfidenceBoundTracker<PolyStitchingConfidenceBoundTracker>);

// UCB is a class that implements the UCB algorithm
// Tracker: maintains confidence bounds per arm
// Reward: (int between 0 and n-1) -> double
template <ConfidenceBoundTracker Tracker, ArmReward Reward>
struct UCB {
  int n;
  double alpha;
  double eps;
  Reward& reward;
  std::vector<Tracker> trackers;
  std::vector<char> enabled;
  bool lazy;
  UCB(int n,
      double alpha,
      double eps,
      Reward& reward,
      std::vector<Tracker> trackers,
      bool lazy = false)
      : n(n),
        alpha(alpha),
        eps(eps),
        reward(reward),
        trackers(std::move(trackers)),
        enabled(n, true),
        lazy(lazy) {
    assert(this->n >= 1);
    assert(this->n == static_cast<int>(this->trackers.size()));
  }

  auto reset() -> void {
    for (auto& tracker : trackers) {
      tracker.reset_ucb();
    }
  }

  auto enable_all_arms() -> void { std::ranges::fill(enabled, true); }

  auto disable_arm(int i) -> void {
    assert(0 <= i && i < n);
    enabled[i] = false;
  }

  [[nodiscard]] auto n_pulls() const -> int {
    return std::ranges::fold_left(
        trackers, 0, [](int total, const auto& tracker) {
          return total + static_cast<int>(tracker.num_pulls());
        });
  }

  [[nodiscard]] auto has_enabled_arm() const -> bool {
    return std::ranges::any_of(enabled, [](char value) { return value; });
  }

  [[nodiscard]] auto best_arm() -> int {
    assert(has_enabled_arm());
    for (int i = 0; i < n; i++) {
      if (!enabled[i]) {
        continue;
      }
      if (trackers[i].num_pulls() == 0) {
        trackers[i].add_sample(reward(i));
      }
    }

    for (size_t t = n;; t++) {
      int j = -1;
      double best_mean = -infty;
      for (int arm = 0; arm < n; arm++) {
        if (!enabled[arm]) {
          continue;
        }
        auto arm_mean = trackers[arm].mean();
        if (j == -1 || arm_mean > best_mean) {
          j = arm;
          best_mean = arm_mean;
        }
      }
      assert(j != -1);
      if (pull(j, t)) {
        my_log(std::format("UCB stops at round {} with arm {}", t, j));
        return j;
      }

      int i = -1;
      double best_ucb = -infty;
      for (int arm = 0; arm < n; arm++) {
        if (!enabled[arm] || arm == j) {
          continue;
        }
        auto arm_ucb = trackers[arm].ucb();
        if (i == -1 || arm_ucb > best_ucb) {
          i = arm;
          best_ucb = arm_ucb;
        }
      }
      if (i == -1) {
        return j;
      }

      if (pull(i, t)) {
        my_log(std::format("UCB stops at round {} with arm {}", t, i));
        return i;
      }
      if (trackers[j].ucb() > trackers[i].ucb()) {
        i = j;
      }

      auto confidence_width = trackers[i].mean() - trackers[i].lcb();
      if (confidence_width < eps) {
        my_log(
            std::format("UCB stops at round {} with arm {} due to {} pulls, "
                        "confidence bound {} < {}, avg reward {}",
                        t, i, trackers[i].num_pulls(), confidence_width, eps,
                        trackers[i].mean()));
        return i;
      }
    }
  }

  [[nodiscard]] auto pull(int i, size_t t) -> bool {
    assert(0 <= i && i < n);
    assert(enabled[i]);
    trackers[i].add_sample(reward(i));

    auto pulls = trackers[i].num_pulls();
    if (!lazy && pulls >= 1 + alpha * (t - pulls)) {
      my_log("Due to num_pulls,");
      return true;
    }

    double my_lower = trackers[i].lcb();
    double max_other_upper = -infty;
    for (int j = 0; j < n; j++) {
      if (j == i || !enabled[j])
        continue;
      max_other_upper = std::max(max_other_upper, trackers[j].ucb());
    }
    if (my_lower > eps + max_other_upper) {
      my_log("Due to confidence bound,");
      return true;
    }
    return false;
  }
};

}  // namespace im

using im::E;
using im::EmpiricalStitchingBoundary;
using im::PolyStitchingConfidenceBoundTracker;
using im::infty;
using im::LILConfidence;
using im::LILConfidenceBoundTracker;
using im::UCB;
