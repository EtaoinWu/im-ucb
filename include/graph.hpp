#pragma once

#include <algorithm>
#include <expected>
#include <istream>
#include <string>
#include <tuple>
#include <vector>

namespace im {

using weight_t = double;
using error_t = std::string;

struct Edge {
  int to;
  weight_t weight;

  [[nodiscard]] friend constexpr bool operator==(const Edge &, const Edge &) =
      default;
};

struct Graph {
  int n;
  int m;
  std::vector<std::vector<Edge>> adj;

  explicit Graph(int n) : n(n), m(0), adj(static_cast<size_t>(n)) {}

  void add_edge(int u, int v, weight_t w) {
    m++;
    adj[u].push_back({v, w});
  }

  void delete_vertex(int u) {
    for (int i = 0; i < n; i++) {
      auto &edges = adj[i];
      auto end_it = std::remove_if(edges.begin(), edges.end(),
                                   [u](const Edge &e) { return e.to == u; });
      m -= edges.end() - end_it;
      edges.erase(end_it, edges.end());
    }
  }

  [[nodiscard]] std::vector<Edge> &operator[](int u) { return adj[u]; }

  [[nodiscard]] const std::vector<Edge> &operator[](int u) const {
    return adj[u];
  }

  [[nodiscard]] std::vector<std::tuple<int, int, weight_t>> get_edges() const;
};

inline std::vector<std::tuple<int, int, weight_t>> Graph::get_edges() const {
  std::vector<std::tuple<int, int, weight_t>> edges;
  for (int u = 0; u < n; ++u) {
    for (const auto &e : adj[u]) {
      edges.emplace_back(u, e.to, e.weight);
    }
  }
  std::ranges::sort(edges);
  return edges;
}

[[nodiscard]] std::expected<Graph, error_t> parse_graph(std::istream &is);
[[nodiscard]] std::expected<Graph, error_t>
load_graph_expected(const std::string &source);

Graph load_graph(const std::string &source);
Graph load_graph(std::istream &is);

} // namespace im

using weight_t = im::weight_t;
using Edge = im::Edge;
using Graph = im::Graph;
using im::load_graph;
using im::load_graph_expected;
using im::parse_graph;
