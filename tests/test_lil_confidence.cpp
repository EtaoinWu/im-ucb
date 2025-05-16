#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
