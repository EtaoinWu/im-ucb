#include <set>
#include <sstream>
#include <tuple>

using std::istringstream;
using std::make_tuple;
using std::set;
using namespace std::literals;

#include <catch2/catch_test_macros.hpp>

#include "graph.hpp"

TEST_CASE("Basic graph operations", "[graph]") {
  Graph g(5);
  g.add_edge(0, 1, 1);
  g.add_edge(0, 2, 2);
  g.add_edge(1, 2, 3);
  g.add_edge(3, 4, 4);

  SECTION("Basic edge access") {
    auto edges = g.get_edges();
    REQUIRE(edges.size() == 4);
    REQUIRE(edges[0] == make_tuple(0, 1, 1.0));
    REQUIRE(edges[1] == make_tuple(0, 2, 2.0));
    REQUIRE(edges[2] == make_tuple(1, 2, 3.0));
    REQUIRE(edges[3] == make_tuple(3, 4, 4.0));
  }

  SECTION("Neighbor access") {
    auto neighbors = g[0];
    int count = 0;
    for (auto e : neighbors) {
      if (e.to == 1) {
        REQUIRE(e.weight == 1);
      } else if (e.to == 2) {
        REQUIRE(e.weight == 2);
      }
      count++;
    }
    REQUIRE(count == 2);
  }
}

TEST_CASE("Graph vertex deletion", "[graph]") {
  auto graph_spec = R"(
    6 9
    0 3 0
    0 4 1
    0 5 2
    1 4 4
    1 5 5
    1 3 3
    2 5 8
    2 4 7
    2 3 6
  )";

  auto iss = istringstream(graph_spec);
  auto g = load_graph(iss);

  SECTION("Delete first vertex") {
    g.delete_vertex(5);

    for (auto u : {0, 1, 2}) {
      set<int> neighbors;
      for (auto e : g[u]) {
        REQUIRE(e.to != 5);
        REQUIRE(neighbors.count(e.to) == 0);
        neighbors.insert(e.to);
      }
      REQUIRE(neighbors.size() == 2);
    }
  }

  SECTION("Delete next vertex") {
    g.delete_vertex(5);
    g.delete_vertex(3);

    for (auto u : {0, 1, 2}) {
      int count = 0;
      for (auto e : g[u]) {
        REQUIRE(e.to == 4);
        count++;
      }
      REQUIRE(count == 1);
    }
  }

  SECTION("Delete last vertex") {
    g.delete_vertex(5);
    g.delete_vertex(3);
    g.delete_vertex(4);

    for (auto u : {0, 1, 2}) {
      for ([[maybe_unused]] auto _ : g[u]) {
        REQUIRE(false); // should not reach here
      }
    }
  }
}