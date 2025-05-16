#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>
#include <format>
using std::accumulate;
using std::exp;
using std::log;
using std::max;
using std::max_element;
using std::min;
using std::sqrt;
using std::vector;

// #include "log.hpp"
#define my_log(x)

const double E = exp(1.0);
// A confidence function is a (double) -> double

struct LILConfidence {
  double mult;
  double logkappap1;
  double logdelta;
  LILConfidence(double kappa, double delta, double sigma)
      : mult((1 + sqrt(kappa)) * sigma * sqrt(2 * (1 + kappa))),
        logkappap1(log(1 + kappa)), logdelta(log(delta)) {
    assert(delta * E < logkappap1);
  }
  double operator()(double t) const {
    return mult * sqrt((log(logkappap1 + log(t)) - logdelta) / t);
  }
};

// UCB is a class that implements the UCB algorithm
// Confidence: (double) -> double
// Reward: (int between 0 and n-1) -> double
template <typename Confidence, typename Reward> struct UCB {
  int n;
  double alpha, beta;
  double eps;
  const Confidence &conf;
  const Reward &reward;
  vector<double> upper_bounds;
  vector<double> sum_rewards;
  vector<size_t> num_pulls;
  vector<double> mean_rewards;
  bool lazy;
  UCB(int n, double alpha, double beta, double eps, const Confidence &conf,
      const Reward &reward, vector<double> upper_bounds = {}, bool lazy = false)
      : n(n), alpha(alpha), beta(beta), eps(eps), conf(conf), reward(reward),
        upper_bounds(upper_bounds), sum_rewards(n, 0), num_pulls(n, 0),
        mean_rewards(n, 0), lazy(lazy) {
    if (this->upper_bounds.empty()) {
      this->upper_bounds.resize(n, INFINITY);
    }
  }
  void reset() { fill(upper_bounds.begin(), upper_bounds.end(), INFINITY); }
  int n_pulls() const {
    return accumulate(num_pulls.begin(), num_pulls.end(), 0);
  }
  int best_arm() {
    for (int i = 0; i < n; i++) {
      num_pulls[i] = 1;
      mean_rewards[i] = sum_rewards[i] = reward(i);
      upper_bounds[i] =
          min(upper_bounds[i],
              sum_rewards[i] / num_pulls[i] + (1 + beta) * conf(num_pulls[i]));
    }
    for (size_t t = n;; t++) {
      auto j = max_element(mean_rewards.begin(), mean_rewards.end()) -
               mean_rewards.begin();
      if (pull(j, t)) {
        my_log(std::format("BAI({}) = {} by best mean", t, j));
        return j;
      }
      auto upper_bounds_copy = upper_bounds[j];
      upper_bounds[j] = -INFINITY;
      auto i = max_element(upper_bounds.begin(), upper_bounds.end()) -
               upper_bounds.begin();
      upper_bounds[j] = upper_bounds_copy;
      if (pull(i, t)) {
        my_log(std::format("BAI({}) = {} by best upper bound", t, i));
        return i;
      }
      if (upper_bounds[j] > upper_bounds[i]) {
        i = j;
      }
      if (conf(num_pulls[i]) < eps) {
        my_log(std::format("BAI({}) = {} by confidence level", t, i));
        return i;
      }
    }
  }

private:
  bool pull(int i, int t) {
    auto r = reward(i);
    sum_rewards[i] += r;
    num_pulls[i]++;
    mean_rewards[i] = sum_rewards[i] / num_pulls[i];
    upper_bounds[i] =
        min(upper_bounds[i], mean_rewards[i] + (1 + beta) * conf(num_pulls[i]));
    if (!lazy && num_pulls[i] >= 1 + alpha * (t - num_pulls[i])) {
      return true;
    }
    double my_lower = mean_rewards[i] - conf(num_pulls[i]);
    double max_other_upper = -INFINITY;
    for (int j = 0; j < n; j++) {
      if (j == i)
        continue;
      max_other_upper =
          max(max_other_upper,
              min(upper_bounds[j], mean_rewards[j] + conf(num_pulls[j])));
    }
    if (my_lower > max_other_upper) {
      return true;
    }
    return false;
  }
};
#undef my_log
