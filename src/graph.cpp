#include <fstream>
#include <format>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "graph.hpp"

namespace im {

auto parse_graph(std::istream &is) -> std::expected<Graph, error_t> {
  int n, m;
  if (!(is >> n >> m)) {
    return std::unexpected("Invalid graph header: expected <n> <m>");
  }
  if (n < 0 || m < 0) {
    return std::unexpected("Invalid graph header: n and m must be non-negative");
  }

  Graph g(n);
  for (int i = 0; i < m; i++) {
    int u, v;
    weight_t w;
    if (!(is >> u >> v >> w)) {
      return std::unexpected(std::format(
          "Invalid edge at index {}: expected <u> <v> <w>", i));
    }
    if (u < 0 || u >= n || v < 0 || v >= n) {
      return std::unexpected(std::format(
          "Invalid edge at index {}: vertex out of range [{}..{})", i, 0, n));
    }
    g.add_edge(u, v, w);
  }
  return g;
}

auto load_graph_expected(std::string_view source)
    -> std::expected<Graph, error_t> {
  if (source.find('\n') != std::string_view::npos) {
    // source is the content of the file
    auto iss = std::istringstream(std::string(source));
    return parse_graph(iss);
  } else {
    // source is a filename
    std::ifstream file{std::string(source)};
    if (!file.is_open()) {
      return std::unexpected(std::format("Failed to open file: {}", source));
    }
    return parse_graph(file);
  }
}

auto load_graph(std::istream &is) -> Graph {
  auto result = parse_graph(is);
  if (!result) {
    throw std::runtime_error(result.error());
  }
  return *std::move(result);
}

auto load_graph(std::string_view source) -> Graph {
  auto result = load_graph_expected(source);
  if (!result) {
    throw std::runtime_error(result.error());
  }
  return *std::move(result);
}

} // namespace im
