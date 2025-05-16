#include <utility>
using std::make_pair;

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

using Catch::Matchers::UnorderedRangeEquals;

#include "diffusion.hpp"
#include "graph.hpp"
#include "greedy.hpp"

TEST_CASE("Greedy on simple functions", "[greedy]") {
  auto marginal = [](auto f) {
    return [f](const vector<int> &S, const vector<int> &A) {
      auto SA = S;
      SA.reserve(S.size() + A.size());
      SA.insert(SA.end(), A.begin(), A.end());
      return f(SA) - f(A);
    };
  };

  auto total = [](const vector<int> &S) {
    vector<double> weights(S.size());
    for (size_t i = 0; i < S.size(); i++) {
      weights[i] = 1. / std::abs(S[i] - 7.3);
    }
    return std::accumulate(weights.begin(), weights.end(), 0.0);
  };
  auto result = greedy_submodular(total, 11, 3);
  REQUIRE_THAT(result, UnorderedRangeEquals({6, 7, 8}));
  result = greedy_lazy_forward(marginal(total), 11, 3);
  REQUIRE_THAT(result, UnorderedRangeEquals({6, 7, 8}));

  auto total_halved = [](vector<int> S) {
    std::sort(S.begin(), S.end());
    double sum = 0;
    for (size_t i = 0; i < S.size(); i++) {
      sum += S[i] * 1.0 / (S.size() - i);
    }
    return sum;
  };
  result = greedy_submodular(total_halved, 10, 4);
  REQUIRE_THAT(result, UnorderedRangeEquals({6, 7, 8, 9}));
  result = greedy_lazy_forward(marginal(total_halved), 10, 4);
  REQUIRE_THAT(result, UnorderedRangeEquals({6, 7, 8, 9}));
}

TEST_CASE("Greedy on a simple graph", "[greedy]") {
  auto edge_weights =
      GENERATE(make_pair(1.0, 0.1), make_pair(0.5, 0.03), make_pair(0.2, 0.03));
  auto [edge_weight, eps] = edge_weights;

  Graph g(6);
  g.add_edge(0, 1, edge_weight);
  g.add_edge(1, 2, edge_weight);
  g.add_edge(3, 5, edge_weight);
  g.add_edge(4, 5, edge_weight);

  SECTION("Greedy") {
    auto greedy = GreedyDiffusion(g, DiffusionType::IndependentCascade, 3, eps,
                                  0.01, greedy_submodular<DiffusionEvaluate>);
    auto result = greedy.run(1);
    CAPTURE(greedy.samples());
    REQUIRE_THAT(result, UnorderedRangeEquals({0, 3, 4}));
  }
}

TEST_CASE("CELF on a simple graph", "[greedy]") {
  auto edge_weights =
      GENERATE(make_pair(1.0, 0.1), make_pair(0.5, 0.03), make_pair(0.2, 0.03));
  auto [edge_weight, eps] = edge_weights;

  Graph g(6);
  g.add_edge(0, 1, edge_weight);
  g.add_edge(1, 2, edge_weight);
  g.add_edge(3, 5, edge_weight);
  g.add_edge(4, 5, edge_weight);

  SECTION("Lazy Forward") {
    auto celf = GreedyDiffusion(g, DiffusionType::IndependentCascade, 3, eps,
                                0.01, greedy_lazy_forward<DiffusionEvaluate>);
    auto celf_result = celf.run(2);
    CAPTURE(celf.samples());
    REQUIRE_THAT(celf_result, UnorderedRangeEquals({0, 3, 4}));
  }
}
