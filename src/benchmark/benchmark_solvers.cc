#include <random>

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "jlp/linear_solver.h"

class LinearProblemFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) {
    // std::tie(m, n, max) = state.range(0);
    m = state.range(0);
    n = state.range(1);
    max = static_cast<double>(state.range(2));
    A.resize(m);
    for (int i = 0; i < m; ++i) A[i].resize(m + n);
    c.resize(m + n);
    b.resize(m);

    std::mt19937 gen(rd());
    // The cost coefficients should range be larger than 0.0, to avoid unbound
    // LPs
    std::uniform_real_distribution<double> dis(0.0, max);

    for (int i = 0; i < n; ++i) c[i] = dis(gen);
    for (int i = n; i < m + n; ++i) c[i] = 0.0;
    for (int i = 0; i < m; ++i) {
      // Avoid 0.0, otherwise the LP problem is most likely fixed
      double tmp = dis(gen);
      while (tmp == 0.0) {
        tmp = dis(gen);
      }
      b[i] = tmp;
    }
    // Identity matrix corresponding to the slack variables
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        A[i][j] = dis(gen);
      }
      for (int j = 0; j < m; ++j) {
        A[i][j] = (i == j) ? 1.0 : 0.0;
      }
    }
  }

  // void SetUp(::benchmark::State& state) { SetUp(state); }

  // void TearDown(::benchmark::State& /* state */) {}

  void TearDown(const ::benchmark::State& /* state */) {}

 protected:
  // The internal data structure for the Linear Problem
  int m, n;
  // Maximum of random floating point number
  double max;
  std::vector<std::vector<double>> A;
  std::vector<double> b, c;
  std::random_device rd;
};

BENCHMARK_DEFINE_F(LinearProblemFixture, BM_DenseJLPSolver)
(benchmark::State& state) {
  using namespace compute_tools;
  // invert to suit our solver for maximization problems
  for (int i = 0; i < n; ++i) c[i] = -c[i];
  ProblemStatus status;
  // double objective_value = 0.0;
  int num_iteration = 1000;
  std::vector<double> basic_solution;
  std::vector<int> basic_indices;
  for (auto _ : state) {
    std::tie(status, basic_solution, basic_indices) =
        Solve(A, b, c, num_iteration);
  }
  state.SetItemsProcessed(int64_t(state.iterations()));
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i = 2; i <= 2048; i <<= 1) b->Args({i, i, 100});
}

BENCHMARK_REGISTER_F(LinearProblemFixture, BM_DenseJLPSolver)
    ->Apply(CustomArguments)
    ->Unit(benchmark::kMillisecond);
