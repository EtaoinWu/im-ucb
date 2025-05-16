#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using std::istream;
using std::remove_if;
using std::sort;
using std::string;
using std::tuple;
using std::vector;

using weight_t = double;

struct Edge {
  int to;
  weight_t weight;
};

struct Graph {
  int n;
  int m;
  vector<vector<Edge>> adj;

  Graph(int n) : n(n), m(0), adj(n, vector<Edge>{}) {}

  void add_edge(int u, int v, weight_t w) {
    m++;
    adj[u].push_back({v, w});
  }

  void delete_vertex(int u) {
    for (int i = 0; i < n; i++) {
      auto &edges = adj[i];
      auto end_it = remove_if(edges.begin(), edges.end(),
                              [u](const Edge &e) { return e.to == u; });
      m -= edges.end() - end_it;
      edges.erase(end_it, edges.end());
    }
  }

  vector<Edge> &operator[](int u) { return adj[u]; }

  const vector<Edge> &operator[](int u) const { return adj[u]; }

  vector<tuple<int, int, weight_t>> get_edges() const;
};

inline vector<tuple<int, int, weight_t>> Graph::get_edges() const {
  Graph &g = const_cast<Graph &>(*this);
  vector<tuple<int, int, weight_t>> edges;
  for (int u = 0; u < n; ++u) {
    for (auto e : g[u]) {
      edges.emplace_back(u, e.to, e.weight);
    }
  }
  sort(edges.begin(), edges.end());
  return edges;
}

Graph load_graph(const string &str);
Graph load_graph(istream &is);
