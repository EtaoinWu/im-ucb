#pragma once

#include <algorithm>
#include <expected>
#include <istream>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace im {

using weight_t = double;
using error_t = std::string;

struct Edge {
  int to;
  weight_t weight;

  [[nodiscard]] friend constexpr bool operator==(const Edge&,
                                                 const Edge&) = default;
};

struct Graph {
  int n;
  int m;
  std::vector<std::vector<Edge>> adj;

  explicit Graph(int n) : n(n), m(0), adj(static_cast<size_t>(n)) {}

  auto add_edge(int u, int v, weight_t w) -> void {
    m++;
    adj[u].push_back({v, w});
  }

  auto delete_vertex(int u) -> void {
    for (int i = 0; i < n; i++) {
      auto& edges = adj[i];
      auto end_it = std::remove_if(edges.begin(), edges.end(),
                                   [u](const Edge& e) { return e.to == u; });
      m -= edges.end() - end_it;
      edges.erase(end_it, edges.end());
    }
  }

  [[nodiscard]] auto operator[](this auto&& self, int u) -> decltype(auto) {
    return std::forward<decltype(self)>(self).adj[u];
  }

  [[nodiscard]] auto get_edges() const
      -> std::vector<std::tuple<int, int, weight_t>>;
};

inline auto Graph::get_edges() const
    -> std::vector<std::tuple<int, int, weight_t>> {
  std::vector<std::tuple<int, int, weight_t>> edges;
  for (int u = 0; u < n; ++u) {
    for (const auto& e : adj[u]) {
      edges.emplace_back(u, e.to, e.weight);
    }
  }
  std::ranges::sort(edges);
  return edges;
}

[[nodiscard]] auto parse_graph(std::istream& is)
    -> std::expected<Graph, error_t>;
[[nodiscard]] auto load_graph_expected(std::string_view source)
    -> std::expected<Graph, error_t>;

[[nodiscard]] auto load_graph(std::string_view source) -> Graph;
[[nodiscard]] auto load_graph(std::istream& is) -> Graph;

}  // namespace im

using weight_t = im::weight_t;
using Edge = im::Edge;
using Graph = im::Graph;
using im::load_graph;
using im::load_graph_expected;
using im::parse_graph;
