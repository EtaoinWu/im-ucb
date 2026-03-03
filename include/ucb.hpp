#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>
#include <vector>

#include "log.hpp"

namespace im {

inline constexpr double E = std::numbers::e_v<double>;
inline constexpr double infty = std::numeric_limits<double>::infinity();

template <typename Fn>
concept ConfidenceRadius = requires(const Fn& fn, size_t t) {
  { fn(t) } -> std::convertible_to<double>;
};

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

struct LILConfidence {
  double mult;
  double logkappap1;
  double logdelta;
  LILConfidence(double kappa, double delta, double sigma)
      : mult((1 + std::sqrt(kappa)) * sigma * std::sqrt(2 * (1 + kappa))),
        logkappap1(std::log(1 + kappa)),
        logdelta(std::log(delta)) {
    assert(delta * E < logkappap1);
  }

  [[nodiscard]] auto operator()(double t) const -> double {
    return mult *
           std::sqrt((std::log(logkappap1 + std::log(t)) - logdelta) / t);
  }
};

static_assert(ConfidenceRadius<LILConfidence>);

struct LILConfidenceBoundTracker {
  double beta;
  LILConfidence radius;
  double range_low;
  double range_high;
  size_t pulls;
  double sum_rewards;
  double mean_reward;
  double capped_ucb;

  LILConfidenceBoundTracker(double kappa = 0.03,
                            double delta = 1e-3,
                            double sigma = 1.0,
                            double beta = 0.5,
                            double range_low = 0.0,
                            double range_high = 1.0)
      : beta(beta),
        radius(kappa, delta, sigma),
        range_low(range_low),
        range_high(range_high),
        pulls(0),
        sum_rewards(0),
        mean_reward(0),
        capped_ucb(infty) {
    assert(range_low <= range_high);
  }

  [[nodiscard]] auto clipped(double value) const -> double {
    return std::clamp(value, range_low, range_high);
  }

  auto add_sample(double sample) -> void {
    auto bounded_sample = clipped(sample);
    pulls++;
    sum_rewards += bounded_sample;
    mean_reward = sum_rewards / pulls;
    auto instant_ucb = clipped(mean_reward + (1 + beta) * radius(pulls));
    capped_ucb = std::min(capped_ucb, instant_ucb);
  }

  auto reset_ucb() -> void { capped_ucb = infty; }

  [[nodiscard]] auto mean() const -> double {
    if (pulls == 0) {
      return (range_low + range_high) / 2;
    }
    return mean_reward;
  }

  [[nodiscard]] auto ucb() const -> double {
    if (pulls == 0) {
      return range_high;
    }
    return clipped(
        std::min(capped_ucb, mean_reward + (1 + beta) * radius(pulls)));
  }

  [[nodiscard]] auto lcb() const -> double {
    if (pulls == 0) {
      return range_low;
    }
    return clipped(mean_reward - radius(pulls));
  }

  [[nodiscard]] auto num_pulls() const -> size_t { return pulls; }
};

static_assert(ConfidenceBoundTracker<LILConfidenceBoundTracker>);

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
using im::infty;
using im::LILConfidence;
using im::LILConfidenceBoundTracker;
using im::UCB;
