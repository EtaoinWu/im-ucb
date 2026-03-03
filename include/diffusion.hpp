#pragma once

#include <cassert>
#include <random>
#include <span>
#include <vector>

#include "graph.hpp"
#include "rng.hpp"

namespace im {

inline std::uniform_real_distribution<double> u01(0.0, 1.0);

enum class DiffusionType {
  IndependentCascade,
  LinearThreshold,
};

struct DiffusionSolver {
  const Graph &g;
  RNG rng;
  size_t times;
  std::vector<size_t> last_activated;
  std::vector<int> queue;
  std::vector<double> weights;
  DiffusionSolver(const Graph &g, seed_type seed)
      : g(g), rng(seed), times(0), last_activated(g.n, 0), queue(g.n, -1),
        weights(g.n, 0.0) {}

  void seed(seed_type seed) { rng.seed(seed); }

private:
  template <typename Iterator>
  [[nodiscard]] Iterator pre_activate(Iterator qr, std::span<const int> origin,
                                      size_t now) {
    for (auto u : origin) {
      if (last_activated[u] < now) {
        last_activated[u] = now;
        *qr++ = u;
      }
    }
    return qr;
  }

  template <typename Iterator>
  [[nodiscard]] Iterator independent_cascade(Iterator ql, Iterator qr,
                                             size_t now) {
    while (ql != qr) {
      int u = *ql++;
      for (const auto &e : g[u]) {
        int v = e.to;
        if (last_activated[v] < now && u01(rng) < e.weight) {
          last_activated[v] = now;
          *qr++ = v;
        }
      }
    }
    return qr;
  }

  template <typename Iterator>
  [[nodiscard]] Iterator linear_threshold(Iterator ql, Iterator qr, size_t now,
                                          size_t then) {
    while (ql != qr) {
      int u = *ql++;
      for (const auto &e : g[u]) {
        int v = e.to;
        if (last_activated[v] < now) {
          last_activated[v] = now;
          weights[v] = u01(rng);
        }
        if (last_activated[v] < then) {
          weights[v] -= e.weight;
          if (weights[v] <= 0) {
            last_activated[v] = then;
            *qr++ = v;
          }
        }
      }
    }
    return qr;
  }

public:
  [[nodiscard]] double run_independent_cascade(
      const std::vector<int> &origin, const std::vector<int> &prepare = {}) {
    auto now = ++times;

    auto ql = queue.data();
    auto qr = queue.data();

    if (!prepare.empty()) {
      qr = pre_activate(qr, std::span<const int>(prepare), now);
      qr = independent_cascade(ql, qr, now);
      ql = qr;
    }

    qr = pre_activate(qr, std::span<const int>(origin), now);
    qr = independent_cascade(ql, qr, now);

    return static_cast<double>(qr - ql);
  }

  [[nodiscard]] double run_linear_threshold(
      const std::vector<int> &origin, const std::vector<int> &prepare = {}) {
    auto now = ++times;
    auto then = ++times;

    auto ql = queue.data();
    auto qr = queue.data();

    if (!prepare.empty()) {
      qr = pre_activate(qr, std::span<const int>(prepare), then);
      qr = linear_threshold(ql, qr, now, then);
      ql = qr;
    }

    qr = pre_activate(qr, std::span<const int>(origin), then);
    qr = linear_threshold(ql, qr, now, then);

    return static_cast<double>(qr - ql);
  }

  [[nodiscard]] double run(DiffusionType type, const std::vector<int> &origin,
                           const std::vector<int> &prepare = {}) {
    switch (type) {
    case DiffusionType::IndependentCascade:
      return run_independent_cascade(origin, prepare);
    case DiffusionType::LinearThreshold:
      return run_linear_threshold(origin, prepare);
    default:
      assert(false);
      return 0.0;
    }
  }
};

} // namespace im

using DiffusionType = im::DiffusionType;
using DiffusionSolver = im::DiffusionSolver;
