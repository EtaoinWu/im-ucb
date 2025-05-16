#include <catch2/catch_test_macros.hpp>

#include "diffusion.hpp"
#include "graph.hpp"

TEST_CASE("Deterministic diffusion on a 2 path graph", "[diffusion_deterministic]") {
  Graph g(6);
  g.add_edge(0, 1, 1);
  g.add_edge(1, 2, 1);
  g.add_edge(2, 3, 1);
  g.add_edge(4, 5, 1);

  DiffusionSolver ds(g, 0);

  SECTION("Independent cascade: simple paths") {
    REQUIRE(ds.run_independent_cascade({0}) == 4);
    REQUIRE(ds.run_independent_cascade({1}) == 3);
    REQUIRE(ds.run_independent_cascade({3}) == 1);
    REQUIRE(ds.run_independent_cascade({4}) == 2);
    REQUIRE(ds.run_independent_cascade({5}) == 1);
    REQUIRE(ds.run_independent_cascade({0, 4}) == 6);
    REQUIRE(ds.run_independent_cascade({1, 5}) == 4);
  }
  SECTION("Linear threshold: simple paths") {
    REQUIRE(ds.run_linear_threshold({0}) == 4);
    REQUIRE(ds.run_linear_threshold({1}) == 3);
    REQUIRE(ds.run_linear_threshold({3}) == 1);
    REQUIRE(ds.run_linear_threshold({4}) == 2);
    REQUIRE(ds.run_linear_threshold({5}) == 1);
  }
  SECTION("Independent cascade: prepare") {
    REQUIRE(ds.run_independent_cascade({0}, {4}) == 4);
    REQUIRE(ds.run_independent_cascade({0}, {1}) == 1);
    REQUIRE(ds.run_independent_cascade({0}, {2}) == 2);
    REQUIRE(ds.run_independent_cascade({4}, {4}) == 0);
  }
  SECTION("Linear threshold: prepare") {
    REQUIRE(ds.run_linear_threshold({0}, {4}) == 4);
    REQUIRE(ds.run_linear_threshold({0}, {1}) == 1);
    REQUIRE(ds.run_linear_threshold({0}, {2}) == 2);
    REQUIRE(ds.run_linear_threshold({4}, {4}) == 0);
  }
  g.add_edge(5, 2, 1);
  SECTION("Independent cascade: joined graph") {
    REQUIRE(ds.run_independent_cascade({0}) == 4);
    REQUIRE(ds.run_independent_cascade({4}) == 4);
    REQUIRE(ds.run_independent_cascade({0, 4}) == 6);
    REQUIRE(ds.run_independent_cascade({0, 4, 1}) == 6);
  }
  SECTION("Linear threshold: joined graph") {
    REQUIRE(ds.run_linear_threshold({0}) == 4);
    REQUIRE(ds.run_linear_threshold({4}) == 4);
    REQUIRE(ds.run_linear_threshold({0, 4}) == 6);
    REQUIRE(ds.run_linear_threshold({0, 4, 1}) == 6);
  }
}
