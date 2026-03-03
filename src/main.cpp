#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include <argparse/argparse.hpp>

#include "cbgreedy.hpp"
#include "diffusion.hpp"
#include "graph.hpp"
#include "greedy.hpp"
#include "log.hpp"

namespace {

using io_result = std::expected<void, std::string>;
using load_result_t = std::expected<std::vector<int>, std::string>;

[[nodiscard]] auto save_result(const std::vector<int>& result,
                               std::string_view dataset,
                               std::string_view alg,
                               int k,
                               const std::vector<size_t>& used_samples)
    -> io_result {
  if (result.size() != used_samples.size()) {
    return std::unexpected("result and used_samples have different size");
  }

  auto dir = std::format("results/{}/{}", dataset, alg);
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }

  auto filename = std::format("{}/{}.txt", dir, k);
  std::ofstream f(filename);
  if (!f.is_open()) {
    return std::unexpected(
        std::format("Failed to open output file {}", filename));
  }

  for (size_t i = 0; i < result.size(); i++) {
    f << result[i] << " " << used_samples[i] << "\n";
  }
  return {};
}

[[nodiscard]] auto load_result(std::string_view dataset,
                               std::string_view alg,
                               int k) -> load_result_t {
  auto filename = std::format("results/{}/{}/{}.txt", dataset, alg, k);
  if (!std::filesystem::exists(filename)) {
    return std::unexpected(std::format("File {} not found", filename));
  }

  std::ifstream f(filename);
  if (!f.is_open()) {
    return std::unexpected(std::format("Failed to open {}", filename));
  }

  std::vector<int> result;
  int v, s;
  while (f >> v >> s) {
    result.push_back(v);
  }

  return result;
}

[[nodiscard]] auto save_eval(std::string_view dataset,
                             std::string_view alg,
                             int k,
                             const std::vector<double>& means) -> io_result {
  auto filename = std::format("results/{}/{}/{}_eval.txt", dataset, alg, k);
  std::ofstream f(filename);
  if (!f.is_open()) {
    return std::unexpected(
        std::format("Failed to open output file {}", filename));
  }

  for (size_t i = 0; i < means.size(); i++) {
    f << means[i] << '\n';
  }

  return {};
}

auto log_io_error(std::string_view context, std::string_view error) -> void {
  std::cerr << context << ": " << error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  argparse::ArgumentParser program("bandit-im");
  program.add_argument("dataset").help("Dataset to use").required();
  program.add_argument("k").help("Run id").required().scan<'i', int>();
  program.add_argument("eps").help("Epsilon").required().scan<'f', double>();
  program.add_argument("delta").help("Delta").required().scan<'f', double>();
  program.add_argument("--n_top")
      .help("Top-k to run")
      .default_value(10)
      .scan<'i', int>();
  program.add_argument("--eval")
      .help("Evaluation mode")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--lt")
      .help("Linear threshold diffusion")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    std::cerr << program;
    return 1;
  }

  auto type = DiffusionType::IndependentCascade;

  auto dataset = program.get<std::string>("dataset");
  auto k = program.get<int>("k");
  auto eps = program.get<double>("eps");
  auto delta = program.get<double>("delta");
  auto n_top = program.get<int>("--n_top");
  auto eval = program.get<bool>("--eval");
  auto lt = program.get<bool>("--lt");
  set_identity(std::format("{} {}", dataset, k));

  if (lt) {
    type = DiffusionType::LinearThreshold;
  }

  auto dataset_path = std::format("data/{}/{}.txt", dataset, dataset);
  if (!std::filesystem::exists(dataset_path)) {
    std::cerr << "Dataset " << dataset << " not found" << '\n';
    return 1;
  }
  auto graph_result = im::load_graph_expected(dataset_path);
  if (!graph_result) {
    std::cerr << "Failed to load graph: " << graph_result.error() << '\n';
    return 1;
  }
  auto g = *std::move(graph_result);

  if (!eval) {
    {
      auto cbgreedy = GreedyCBDiffusion(g, type, n_top, eps, delta,
                                        greedy_cb<DiffusionReward>);
      auto result = cbgreedy.run(10 * k + 3);
      auto saved =
          save_result(result, dataset, "greedy-cb", k, cbgreedy.used_samples());
      if (!saved) {
        log_io_error("Failed to save greedy-cb", saved.error());
      }
    }

    {
      auto celf_cb = GreedyCBDiffusion(g, type, n_top, eps, delta,
                                       greedy_cb_lazy<DiffusionReward>);
      auto celf_result = celf_cb.run(10 * k + 4);
      auto saved = save_result(celf_result, dataset, "celf-cb", k,
                               celf_cb.used_samples());
      if (!saved) {
        log_io_error("Failed to save celf-cb", saved.error());
      }
    }

    {
      auto celf = DiffusionAlgoRun(g, type, n_top, eps, delta,
                                   greedy_lazy_forward<DiffusionSubmodular>);
      auto result = celf.run(10 * k + 2);
      auto saved = save_result(result, dataset, "celf", k, celf.used_samples());
      if (!saved) {
        log_io_error("Failed to save celf", saved.error());
      }
    }

    if (g.n <= 40) {
      auto greedy = DiffusionAlgoRun(g, type, n_top, eps, delta,
                                     greedy_submodular<DiffusionSubmodular>);
      auto result = greedy.run(10 * k + 1);
      auto saved =
          save_result(result, dataset, "greedy", k, greedy.used_samples());
      if (!saved) {
        log_io_error("Failed to save greedy", saved.error());
      }
    }
  } else {
    auto solver = DiffusionSolver(g, k);

    auto evaluate = [&](const std::vector<int>& S) -> double {
      auto total = 0.0, total_sq = 0.0;
      size_t cnt = 0;
      size_t last_cnt = 0;
      while (true) {
        auto result = solver.run(type, S);
        total += result;
        total_sq += result * result;
        if (cnt > 100) {
          auto mean = total / cnt;
          auto var = total_sq / cnt - mean * mean;
          auto std = std::sqrt(var * cnt / (cnt - 1));
          // if the samples really follow a normal distribution,
          // what's our confidence interval on mean?
          auto confidence = 1.96 * std / std::sqrt(cnt);
          if (cnt > last_cnt * 3.1622) {
            last_cnt = cnt;
          }
          if (confidence < eps || confidence < eps * mean) {
            break;
          }
        }
        cnt++;
      }
      return total / cnt;
    };

    for (std::string_view alg : {"greedy-cb", "celf-cb", "celf", "greedy"}) {
      auto result = load_result(dataset, alg, k);
      if (!result) {
        std::cerr << "Result for " << alg << " " << k
                  << " not available: " << result.error() << '\n';
        continue;
      }
      std::vector<double> means;
      for (size_t i = 0; i < result->size(); i++) {
        auto pref = std::vector<int>(result->begin(), result->begin() + i + 1);
        std::cout << std::format("Evaluating {}/{}/{}: size {}", dataset, alg,
                                 k, pref.size())
                  << '\n';
        auto mean = evaluate(pref);
        means.push_back(mean);
      }
      auto saved = save_eval(dataset, alg, k, means);
      if (!saved) {
        log_io_error(std::format("Failed to save evaluation for {}", alg),
                     saved.error());
      }
    }
  }

  return 0;
}
