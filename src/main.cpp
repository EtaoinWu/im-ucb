#include <filesystem>
#include <format>
#include <fstream>
#include <string_view>
#include <vector>

using str = std::string_view;
using std::format;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

#include <argparse/argparse.hpp>

#include "cbgreedy.hpp"
#include "diffusion.hpp"
#include "graph.hpp"
#include "greedy.hpp"
#include "log.hpp"

void save_result(const vector<int> &result, str dataset, str alg, int k,
                 vector<size_t> used_samples) {
  auto dir = format("results/{}/{}", dataset, alg);
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  auto filename = format("{}/{}.txt", dir, k);
  ofstream f(filename);
  if (result.size() != used_samples.size()) {
    std::cerr << "result and used_samples have different size!" << std::endl;
    return;
  }
  for (size_t i = 0; i < result.size(); i++) {
    f << result[i] << " " << used_samples[i] << "\n";
  }
  f.close();
}

vector<int> load_result(str dataset, str alg, int k) {
  auto filename = format("results/{}/{}/{}.txt", dataset, alg, k);
  if (!std::filesystem::exists(filename)) {
    std::cerr << "File " << filename << " not found" << std::endl;
    return vector<int>();
  }
  ifstream f(filename);
  vector<int> result;
  int v, s;
  while (f >> v >> s) {
    result.push_back(v);
  }
  return result;
}

void save_eval(str dataset, str alg, int k, vector<double> means) {
  auto filename = format("results/{}/{}/{}_eval.txt", dataset, alg, k);
  ofstream f(filename);
  for (size_t i = 0; i < means.size(); i++) {
    f << means[i] << std::endl;
  }
  f.close();
}

int main(int argc, char **argv) {
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
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  auto type = DiffusionType::IndependentCascade;

  auto dataset = program.get<string>("dataset");
  auto k = program.get<int>("k");
  auto eps = program.get<double>("eps");
  auto delta = program.get<double>("delta");
  auto n_top = program.get<int>("--n_top");
  auto eval = program.get<bool>("--eval");
  auto lt = program.get<bool>("--lt");
  set_identity(format("{} {}", dataset, k));

  if (lt) {
    type = DiffusionType::LinearThreshold;
  }

  auto dataset_path = format("data/{}/{}.txt", dataset, dataset);
  if (!std::filesystem::exists(dataset_path)) {
    std::cerr << "Dataset " << dataset << " not found" << std::endl;
    return 1;
  }
  auto g = load_graph(dataset_path);

  if (!eval) {
    {
      auto cbgreedy =
          GreedyCBDiffusion(g, type, n_top, eps,
                            delta, greedy_cb<DiffusionReward>);
      auto result = cbgreedy.run(10 * k + 3);
      save_result(result, dataset, "greedy-cb", k, cbgreedy.used_samples());
    }

    {
      auto celf_cb =
          GreedyCBDiffusion(g, type, n_top, eps,
                            delta, greedy_cb_lazy<DiffusionReward>);
      auto celf_result = celf_cb.run(10 * k + 4);
      save_result(celf_result, dataset, "celf-cb", k, celf_cb.used_samples());
    }

    {
      auto celf =
          GreedyDiffusion(g, type, n_top, eps,
                          delta, greedy_lazy_forward<DiffusionEvaluate>);
      auto result = celf.run(10 * k + 2);
      save_result(result, dataset, "celf", k, celf.used_samples());
    }

    if (g.n <= 40) {
      auto greedy =
          GreedyDiffusion(g, type, n_top, eps,
                          delta, greedy_submodular<DiffusionEvaluate>);
      auto result = greedy.run(10 * k + 1);
      save_result(result, dataset, "greedy", k, greedy.used_samples());
    }
  } else {
    auto solver = DiffusionSolver(g, k);

    auto evaluate = [&](const vector<int> &S) {
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
          auto std = sqrt(var * cnt / (cnt - 1));
          // if the samples really follow a normal distribution,
          // what's our confidence interval on mean?
          auto confidence = 1.96 * std / sqrt(cnt);
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

    for (auto alg : {"greedy-cb", "celf-cb", "celf", "greedy"}) {
      auto result = load_result(dataset, alg, k);
      if (result.empty()) {
        std::cerr << "Result for " << alg << " " << k << " not found"
                  << std::endl;
        continue;
      }
      vector<double> means;
      for (size_t i = 0; i < result.size(); i++) {
        auto pref = vector<int>(result.begin(), result.begin() + i + 1);
        std::cout << format("Evaluating {}/{}/{}: size {}", dataset, alg, k,
                            pref.size())
                  << std::endl;
        auto mean = evaluate(pref);
        means.push_back(mean);
      }
      save_eval(dataset, alg, k, means);
    }
  }

  return 0;
}
