#include <utility>
using std::make_pair;

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

using Catch::Matchers::UnorderedRangeEquals;

#include "diffusion.hpp"
#include "graph.hpp"
#include "cbgreedy.hpp"

TEST_CASE("Confidence-based Greedy on a simple graph", "[greedy]") {
  auto edge_weights =
      GENERATE(make_pair(1.0, 0.1), make_pair(0.5, 0.03), make_pair(0.2, 0.03));
  auto [edge_weight, eps] = edge_weights;

  Graph g(6);
  g.add_edge(0, 1, edge_weight);
  g.add_edge(1, 2, edge_weight);
  g.add_edge(3, 5, edge_weight);
  g.add_edge(4, 5, edge_weight);

  SECTION("Greedy") {
    auto gcb = GreedyCBDiffusion(g, DiffusionType::IndependentCascade, 3, eps,
                                  0.01, greedy_cb<DiffusionReward>);
    auto result = gcb.run(1);
    CAPTURE(gcb.samples());
    REQUIRE_THAT(result, UnorderedRangeEquals({0, 3, 4}));
  }
}

TEST_CASE("Confidence-based CELF on a simple graph", "[greedy]") {
  auto edge_weights =
      GENERATE(make_pair(1.0, 0.1), make_pair(0.5, 0.03), make_pair(0.2, 0.03));
  auto [edge_weight, eps] = edge_weights;

  Graph g(6);
  g.add_edge(0, 1, edge_weight);
  g.add_edge(1, 2, edge_weight);
  g.add_edge(3, 5, edge_weight);
  g.add_edge(4, 5, edge_weight);

  SECTION("Lazy Forward") {
    auto celf_cb = GreedyCBDiffusion(g, DiffusionType::IndependentCascade, 3, eps,
                                0.01, greedy_cb_lazy<DiffusionReward>);
    auto celf_result = celf_cb.run(2);
    CAPTURE(celf_cb.samples());
    REQUIRE_THAT(celf_result, UnorderedRangeEquals({0, 3, 4}));
  }
}
