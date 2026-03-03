#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <format>
#include <iostream>
#include <random>

using Catch::Matchers::WithinRel;

#include "ucb.hpp"

TEST_CASE("LILConfidence", "[ucb]") {
  auto kappa = GENERATE(0.01, 0.03, 0.1);
  auto delta_kappa = GENERATE(0.1, 0.2, 0.4, 0.8);
  auto delta = delta_kappa * kappa / E;
  auto sigma = GENERATE(0.5, 1.0, 16.0);
  auto lil = LILConfidence(kappa, delta, sigma);
  auto t = GENERATE(1, 2, 6, 177, 114514);
  CAPTURE(kappa, delta, sigma, t);
  auto lil_t = lil(t);
  //  U_{\kappa,\delta,\sigma}(T) =
  // (1+\sqrt{\kappa}) \dot
  // \sqrt{\frac{2\sigma^2 (1+\kappa)}{T} \log
  // \frac{\log\rbr{(1+\kappa)T}}{\delta}}.
  auto reference =
      (1 + sqrt(kappa)) * sqrt(2 * sigma * sigma * (1 + kappa) / t *
                               log(log((1 + kappa) * t) / delta));
  REQUIRE_THAT(lil_t, WithinRel(reference, 1e-6));
}

TEST_CASE("Empirical variance stitching is anytime-valid in simulation",
          "[ucb][variance][slow]") {
  constexpr int trials = 10000;
  constexpr int horizon = 1000;
  constexpr double alpha = 0.1;

  auto run_anytime_check = [&](double p, uint64_t seed) {
    std::mt19937_64 engine(seed);
    std::bernoulli_distribution distribution(p);
    int failures = 0;
    for (int trial = 0; trial < trials; trial++) {
      auto tracker = PolyStitchingConfidenceBoundTracker(
          alpha, 2.280, 1.0, 1.418, 0.0, 1.0);
      auto violated = false;
      for (int t = 1; t <= horizon; t++) {
        auto x = distribution(engine) ? 1.0 : 0.0;
        tracker.add_sample(x);
        if (!(tracker.lcb() < p && p < tracker.ucb())) {
          violated = true;
          break;
        }
      }
      if (violated) {
        failures++;
      }
    }
    auto failure_rate = static_cast<double>(failures) / static_cast<double>(trials);
    std::cout << std::format(
                     "EmpiricalStitching anytime failure rate for Bernoulli({}): {}%\n",
                     p, 100.0 * failure_rate)
              << std::flush;
    return failure_rate;
  };

  auto failure_rate_half = run_anytime_check(0.5, 0xA511CE55ULL);
  auto failure_rate_sparse = run_anytime_check(0.01, 0x5EED0011ULL);

  REQUIRE(failure_rate_half <= 0.13);
  REQUIRE(failure_rate_sparse <= 0.13);
}
