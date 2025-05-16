#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "ucb.hpp"
#include "rng.hpp"
using std::normal_distribution;

struct GaussianReward {
  mutable RNG rng;
  vector<double> means;
  double sigma;
  GaussianReward(vector<double> means, double sigma, seed_type seed): rng(seed), means(means), sigma(sigma) {}
  double operator()(int i) const {
    return normal_distribution<double>(means[i], sigma)(rng);
  }
};

TEST_CASE("UCB with Gaussian rewards", "[ucb]") {
  auto n = GENERATE(3, 6, 12);
  vector<double> means(n);
  for (int i = 0; i < n; i++) {
    means[i] = i * 1.0 / n;
  }
  auto sigma = 1.0;
  auto seed = 1234567890 + n;
  auto reward = GaussianReward(means, sigma, seed);
  auto lil = LILConfidence(0.01, 0.001, 2 * sigma);
  auto ucb = UCB(n, 3.0, 1.0, 0.03 / n, lil, reward);
  auto best_arm = ucb.best_arm();
  CAPTURE(n, ucb.n_pulls());
  REQUIRE(best_arm == n - 1);
}
