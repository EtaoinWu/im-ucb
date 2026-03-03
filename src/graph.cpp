#include <fstream>
#include <format>
#include <sstream>
#include <string>
#include <stdexcept>

#include "graph.hpp"

namespace im {

std::expected<Graph, error_t> parse_graph(std::istream &is) {
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

std::expected<Graph, error_t> load_graph_expected(const std::string &source) {
  if (source.find('\n') != std::string::npos) {
    // s is the content of the file
    auto iss = std::istringstream(source);
    return parse_graph(iss);
  } else {
    // s is a filename
    auto filename = source;
    std::ifstream file(filename);
    if (!file.is_open()) {
      return std::unexpected("Failed to open file: " + filename);
    }
    return parse_graph(file);
  }
}

Graph load_graph(std::istream &is) {
  auto result = parse_graph(is);
  if (!result) {
    throw std::runtime_error(result.error());
  }
  return std::move(result.value());
}

Graph load_graph(const std::string &source) {
  auto result = load_graph_expected(source);
  if (!result) {
    throw std::runtime_error(result.error());
  }
  return std::move(result.value());
}

} // namespace im
