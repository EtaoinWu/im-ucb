#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

#include "diffusion.hpp"
#include "graph.hpp"
#include "utility.hpp"

TEST_CASE("Diffusion on a path graph", "[diffusion_minimal]") {
  Graph g(6);
  g.add_edge(0, 1, 0.5);
  g.add_edge(1, 2, 0.5);
  g.add_edge(2, 3, 0.5);
  g.add_edge(3, 4, 0.5);
  g.add_edge(4, 5, 0.5);

  DiffusionSolver ds(g, 0);

  SECTION("Independent cascade") {
    auto results = repeat_avg(10000, [&]() {
      return ds.run_independent_cascade({4});
    });
    REQUIRE_THAT(results, WithinAbs(1.50, 0.03));
    results = repeat_avg(10000, [&]() {
      return ds.run_independent_cascade({0});
    });
    REQUIRE_THAT(results, WithinAbs(2.0 - 1.0 / 32, 0.03));
    results = repeat_avg(10000, [&]() {
      return ds.run_independent_cascade({0}, {1});
    });
    REQUIRE(results == 1.0);
  }

  SECTION("Linear threshold") {
    auto results = repeat_avg(10000, [&]() {
      return ds.run_linear_threshold({4});
    });
    REQUIRE_THAT(results, WithinAbs(1.50, 0.03));
    results = repeat_avg(10000, [&]() {
      return ds.run_linear_threshold({0});
    });
    REQUIRE_THAT(results, WithinAbs(2.0 - 1.0 / 32, 0.03));
    results = repeat_avg(10000, [&]() {
      return ds.run_linear_threshold({0}, {1});
    });
    REQUIRE(results == 1.0);
  }
}
