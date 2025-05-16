#pragma once

#include <cassert>
#include <random>
#include <variant>
#include <vector>

using std::variant;
using std::vector;

#include "graph.hpp"
#include "rng.hpp"

static std::uniform_real_distribution<double> u01(0.0, 1.0);

enum class DiffusionType {
  IndependentCascade,
  LinearThreshold,
};

struct DiffusionSolver {
  const Graph &g;
  RNG rng;
  size_t times;
  vector<size_t> last_activated;
  vector<int> queue;
  vector<double> weights;
  DiffusionSolver(const Graph &g, seed_type seed)
      : g(g), rng(seed), times(0), last_activated(g.n, 0), queue(g.n, -1),
        weights(g.n, 0.0) {}

  void seed(seed_type seed) { rng.seed(seed); }

private:
  template <typename iterator>
  [[clang::always_inline]] iterator
  pre_activate(iterator qr, const vector<int> &origin, size_t now) {
    for (auto u : origin) {
      if (last_activated[u] < now) {
        last_activated[u] = now;
        *qr++ = u;
      }
    }
    return qr;
  }

  template <typename iterator>
  [[clang::always_inline]] iterator
  independent_cascade(iterator ql, iterator qr, size_t now) {
    while (ql != qr) {
      int u = *ql++;
      for (auto e : g[u]) {
        int v = e.to;
        if (last_activated[v] < now && u01(rng) < e.weight) {
          last_activated[v] = now;
          *qr++ = v;
        }
      }
    }
    return qr;
  }
  template <typename iterator>
  [[clang::always_inline]] iterator linear_threshold(iterator ql, iterator qr,
                                                     size_t now, size_t then) {
    while (ql != qr) {
      int u = *ql++;
      for (auto e : g[u]) {
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
  [[clang::always_inline]] double
  run_independent_cascade(const vector<int> &origin,
                          const vector<int> &prepare = {}) {
    auto now = ++times;

    auto ql = queue.data();
    auto qr = queue.data();

    if (!prepare.empty()) {
      qr = pre_activate(qr, prepare, now);
      qr = independent_cascade(ql, qr, now);
      ql = qr;
    }

    qr = pre_activate(qr, origin, now);
    qr = independent_cascade(ql, qr, now);

    return qr - ql;
  }
  [[clang::always_inline]] double
  run_linear_threshold(const vector<int> &origin,
                       const vector<int> &prepare = {}) {
    auto now = ++times;
    auto then = ++times;

    auto ql = queue.data();
    auto qr = queue.data();

    if (!prepare.empty()) {
      qr = pre_activate(qr, prepare, then);
      qr = linear_threshold(ql, qr, now, then);
      ql = qr;
    }

    qr = pre_activate(qr, origin, then);
    qr = linear_threshold(ql, qr, now, then);

    return qr - ql;
  }

  [[clang::always_inline]] double run(DiffusionType type,
                                      const vector<int> &origin,
                                      const vector<int> &prepare = {}) {
    switch (type) {
    case DiffusionType::IndependentCascade:
      return run_independent_cascade(origin, prepare);
    case DiffusionType::LinearThreshold:
      return run_linear_threshold(origin, prepare);
    default:
      assert(false);
    }
  }
};
