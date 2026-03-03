#pragma once

#include <cassert>
#include <initializer_list>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "graph.hpp"
#include "rng.hpp"

namespace im {

inline std::uniform_real_distribution<double> u01(0.0, 1.0);

enum class DiffusionType {
  IndependentCascade,
  LinearThreshold,
};

// The "raw" diffusion calculation logic
// for both IC and LT models
struct DiffusionSolver {
  const Graph& g;
  RNG rng;
  size_t times;
  std::vector<size_t> last_activated;
  std::vector<int> queue;
  std::vector<double> weights;
  DiffusionSolver(const Graph& g, seed_type seed)
      : g(g),
        rng(seed),
        times(0),
        last_activated(g.n, 0),
        queue(g.n, -1),
        weights(g.n, 0.0) {}

  auto seed(seed_type seed) -> void { rng.seed(seed); }

 private:
  [[nodiscard]] auto pre_activate(int* qr,
                                  std::span<const int> origin,
                                  size_t now) -> int* {
    for (auto u : origin) {
      if (last_activated[u] < now) {
        last_activated[u] = now;
        *qr++ = u;
      }
    }
    return qr;
  }

  [[nodiscard]] auto independent_cascade(int* ql, int* qr, size_t now) -> int* {
    while (ql != qr) {
      int u = *ql++;
      for (const auto& e : g[u]) {
        int v = e.to;
        if (last_activated[v] < now && u01(rng) < e.weight) {
          last_activated[v] = now;
          *qr++ = v;
        }
      }
    }
    return qr;
  }

  [[nodiscard]] auto linear_threshold(int* ql, int* qr, size_t now, size_t then)
      -> int* {
    while (ql != qr) {
      int u = *ql++;
      for (const auto& e : g[u]) {
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
  [[nodiscard]] auto run_independent_cascade(std::span<const int> origin,
                                             std::span<const int> prepare = {})
      -> double {
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

    return static_cast<double>(qr - ql);
  }

  [[nodiscard]] auto run_linear_threshold(std::span<const int> origin,
                                          std::span<const int> prepare = {})
      -> double {
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

    return static_cast<double>(qr - ql);
  }

  // workaround: for some reason `initializer_list`s are not `span`s by default
  [[nodiscard]] auto run_independent_cascade(
      std::initializer_list<int> origin,
      std::initializer_list<int> prepare = {}) -> double {
    return run_independent_cascade(std::span<const int>(origin),
                                   std::span<const int>(prepare));
  }

  [[nodiscard]] auto run_linear_threshold(
      std::initializer_list<int> origin,
      std::initializer_list<int> prepare = {}) -> double {
    return run_linear_threshold(std::span<const int>(origin),
                                std::span<const int>(prepare));
  }

  [[nodiscard]] auto run(DiffusionType type,
                         std::span<const int> origin,
                         std::span<const int> prepare = {}) -> double {
    switch (type) {
      case DiffusionType::IndependentCascade:
        return run_independent_cascade(origin, prepare);
      case DiffusionType::LinearThreshold:
        return run_linear_threshold(origin, prepare);
    }
    std::unreachable();
  }

  [[nodiscard]] auto run(DiffusionType type,
                         std::initializer_list<int> origin,
                         std::initializer_list<int> prepare = {}) -> double {
    return run(type, std::span<const int>(origin),
               std::span<const int>(prepare));
  }

  [[nodiscard]] auto run(DiffusionType type,
                         int origin,
                         std::span<const int> prepare = {}) -> double {
    return run(type, std::span<const int>(&origin, 1), prepare);
  }
};

}  // namespace im

using DiffusionType = im::DiffusionType;
using DiffusionSolver = im::DiffusionSolver;
