#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>

#include "graph.hpp"

using std::string;
using std::ifstream;
using std::istringstream;

Graph load_graph(istream &is) {
  int n, m;
  is >> n >> m;
  Graph g(n);
  for (int i = 0; i < m; i++) {
    int u, v;
    weight_t w;
    is >> u >> v >> w;
    g.add_edge(u, v, w);
  }
  return g;
}

Graph load_graph(const string &str) {
  if (str.find("\n") != string::npos) {
    // s is the content of the file
    auto iss = istringstream(str);
    return load_graph(iss);
  } else {
    // s is a filename
    auto filename = str;
    ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    return load_graph(file);
  }
}
