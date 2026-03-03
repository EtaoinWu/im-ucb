#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>

namespace im {

inline constexpr double E = std::numbers::e_v<double>;
inline constexpr double infty = std::numeric_limits<double>::max();

template <typename Fn>
concept ConfidenceRadius = requires(const Fn& fn, size_t t) {
  { fn(t) } -> std::convertible_to<double>;
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

struct EmpiricalStitchingBoundary {
  double eta;
  double m;
  double s;
  double c;
  double k1;
  double k2;
  double log_eta;
  double log_zeta;

  explicit EmpiricalStitchingBoundary(double eta = 2.280,
                                      double m = 1.0,
                                      double s = 1.418,
                                      double c = 1.0)
      : eta(eta),
        m(m),
        s(s),
        c(c),
        k1((std::pow(eta, 0.25) + std::pow(eta, -0.25)) / std::sqrt(2.0)),
        k2((std::sqrt(eta) + 1.0) / 2.0),
        log_eta(std::log(eta)),
        log_zeta(std::log(zeta(s))) {
    assert(this->eta > 1.0);
    assert(this->m > 0.0);
    assert(this->s > 1.0);
    assert(this->c >= 0.0);
  }

  [[nodiscard]] static auto zeta(double exponent) -> double {
#if defined(__cpp_lib_math_special_functions)
    return std::riemann_zeta(exponent);
#else
    constexpr int max_terms = 200000;
    auto sum = 0.0;
    for (int k = 1; k <= max_terms; k++) {
      sum += std::pow(static_cast<double>(k), -exponent);
    }
    return sum;
#endif
  }

  [[nodiscard]] auto epoch(double variance_proxy) const -> size_t {
    if (variance_proxy <= m) {
      return 0;
    }
    auto ratio = variance_proxy / m;
    auto value = std::log(ratio) / log_eta;
    if (value < 0.0) {
      return 0;
    }
    return static_cast<size_t>(std::floor(value));
  }

  [[nodiscard]] auto log_factor(double alpha, double variance_proxy) const
      -> double {
    assert(alpha > 0.0 && alpha < 1.0);
    auto k = static_cast<double>(epoch(variance_proxy));
    auto value = std::log(1.0 / alpha) + log_zeta + s * std::log1p(k);
    return std::max(value, 1e-12);
  }

  [[nodiscard]] auto stitching(double alpha, double variance_proxy) const
      -> double {
    auto ell = log_factor(alpha, variance_proxy);
    auto term = k1 * k1 * variance_proxy * ell + k2 * k2 * c * c * ell * ell;
    return std::sqrt(std::max(0.0, term)) + k2 * c * ell;
  }

  [[nodiscard]] auto boundary(double alpha, double variance_proxy) const
      -> double {
    return stitching(alpha, std::max(variance_proxy, m));
  }

  [[nodiscard]] auto biboundary(double alpha, double variance_proxy) const
      -> double {
    return boundary(alpha / 2.0, variance_proxy);
  }
};

struct PolyStitchingConfidenceBoundTracker {
  double delta;
  EmpiricalStitchingBoundary boundary_fn;
  double range_low;
  double range_high;
  size_t pulls;
  double sum_rewards;
  double mean_reward;
  double empirical_variance;
  double capped_ucb;

  PolyStitchingConfidenceBoundTracker(double delta = 1e-3,
                                           double eta = 2.280,
                                           double m = 1.0,
                                           double s = 1.418,
                                           double range_low = 0.0,
                                           double range_high = 1.0)
      : delta(delta),
        boundary_fn(eta, m, s, range_high - range_low),
        range_low(range_low),
        range_high(range_high),
        pulls(0),
        sum_rewards(0),
        mean_reward(0),
        empirical_variance(0),
        capped_ucb(infty) {
    assert(delta > 0.0 && delta < 1.0);
    assert(range_low <= range_high);
  }

  [[nodiscard]] auto clipped(double value) const -> double {
    return std::clamp(value, range_low, range_high);
  }

  [[nodiscard]] auto radius() const -> double {
    if (pulls == 0) {
      return range_high - range_low;
    }
    return boundary_fn.biboundary(delta, empirical_variance) /
           static_cast<double>(pulls);
  }

  auto add_sample(double sample) -> void {
    auto bounded_sample = clipped(sample);
    if (pulls > 0) {
      auto innovation = bounded_sample - mean_reward;
      empirical_variance += innovation * innovation;
    }
    pulls++;
    sum_rewards += bounded_sample;
    mean_reward = sum_rewards / pulls;
    auto instant_ucb = clipped(mean_reward + radius());
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
    return clipped(std::min(capped_ucb, mean_reward + radius()));
  }

  [[nodiscard]] auto lcb() const -> double {
    if (pulls == 0) {
      return range_low;
    }
    return clipped(mean_reward - radius());
  }

  [[nodiscard]] auto num_pulls() const -> size_t { return pulls; }

  [[nodiscard]] auto empirical_variance_proxy() const -> double {
    return empirical_variance;
  }
};

}  // namespace im

using im::E;
using im::EmpiricalStitchingBoundary;
using im::PolyStitchingConfidenceBoundTracker;
using im::infty;
using im::LILConfidence;
using im::LILConfidenceBoundTracker;
