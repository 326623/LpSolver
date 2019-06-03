#include <cassert>
#include <iostream>
#include <limits>
#include <vector>
#include <tuple>

#include <glog/logging.h>

#include "jlp/linear_solver.h"
#include "jlp/status.h"

namespace compute_tools {
std::tuple<ProblemStatus, std::vector<double>, std::vector<int>> Solve(
    const std::vector<std::vector<double>>& A, const std::vector<double>& b,
    std::vector<double>& c, int num_iterations) {
  int m = A.size();
  DCHECK(m > 0) << "m == 0";
  int n = A.front().size();
  DCHECK(n > 0 && n >= m) << "m == 0 or n == 0 or n < m";

  // Initialization
  std::vector<std::vector<double>> inverse_B(m, std::vector<double>(m, 0));
  std::vector<int> basic_indices(m);
  std::vector<int> nonbasic_indices(n - m);
  std::vector<double> basic_coefficients(m);
  std::vector<double> basic_solutions(m);

  // Treats the slack variables as initail basis
  // Assumes A is passed with slack variables in the back. And the submatrix of
  // slack variables is an identity matrix.
  // The slack variables' coefficients are all zero, therefore the objective
  // value is zero initially.
  for (int i = 0; i < m; ++i) inverse_B[i][i] = 1.0;

  for (int offset = n - m, i = offset; i < n; ++i) {
    basic_coefficients[i - offset] = c[i];
    basic_indices[i - offset] = i;
    basic_solutions[i - offset] = b[i - offset];
  }

  // for (int i = 0, offset = n - m; i < m; ++i) {
  //   basic_coefficients[i] = c[i + offset];
  //   basic_indices[i] = i + offset;
  //   basic_solutions[i] = b[i];
  // }

  for (int i = 0; i < n - m; ++i) nonbasic_indices[i] = i;

  // Intermediate variables

  // For price out
  std::vector<double> simplex_multiplier(m);
  std::vector<double> exchange_reduction(m);
  // For the update of inverse B
  std::vector<double> tmp_column(m);
  std::vector<double> eta(m);
  // initial objective value
  double objective_value = 0.0;

  for (int iteration_pos = 0; iteration_pos < num_iterations; ++iteration_pos) {
    // Compute simplex multiplier
    for (int i = 0; i < m; ++i) {
      simplex_multiplier[i] = 0.0;
      for (int j = 0; j < m; ++j) {
        simplex_multiplier[i] += basic_coefficients[j] * inverse_B[j][i];
      }
    }

    // Pricing, choose the what gives the lowest
    int entering_index = -1;
    int nonbasic_picked = -1;
    // Any positive value that is the lowest means it is optimal
    double best_entering_cost = 1.0;
    // for (int index : nonbasic_indices) {
    for (int i = 0; i < n - m; ++i) {
      int index = nonbasic_indices[i];
      double entering_cost = c[index];
      for (int j = 0; j < m; ++j) {
        entering_cost -= simplex_multiplier[j] * A[j][index];
      }
      if (entering_cost < best_entering_cost) {
        best_entering_cost = entering_cost;
        entering_index = index;
        nonbasic_picked = i;
      }
    }

    // WARNING: Please be careful with floating point issues
    if (best_entering_cost >= 0 || entering_index == -1) {
      // Optimal
      return {ProblemStatus::OPTIMAL, basic_solutions, basic_indices};
    }

    for (int i = 0; i < m; ++i) {
      exchange_reduction[i] = 0.0;
      for (int j = 0; j < m; ++j) {
        exchange_reduction[i] += inverse_B[i][j] * A[j][entering_index];
      }
    }

    // Ratio test
    // Collect the minimal positive of ratio test
    int leaving_index = -1;
    double minimal_ratio_test = std::numeric_limits<double>::max();
    for (int i = 0; i < m; ++i) {
      if (exchange_reduction[i] > 0) {
        double ratio_test = basic_solutions[i] / exchange_reduction[i];
        if (ratio_test < minimal_ratio_test) {
          leaving_index = i;
          minimal_ratio_test = ratio_test;
        }
      }
    }

    if (leaving_index == -1) {
      // Unbounded
      return {ProblemStatus::UNBOUND, basic_solutions, basic_indices};
    }

    // Need to compute a column \eta of the elementary matrix that is used to
    // update the inverse of B for the next iteration
    double alpha_leaving = exchange_reduction[leaving_index];
    for (int i = 0; i < m; ++i) {
      eta[i] = -exchange_reduction[i] / alpha_leaving;
    }
    eta[leaving_index] = 1 / alpha_leaving;

    // Update
    nonbasic_indices[nonbasic_picked] = basic_indices[leaving_index];
    basic_indices[leaving_index] = entering_index;

    // Update inverse_B
    for (int col = 0; col < m; ++col) {
      for (int row = 0; row < m; ++row) {
        tmp_column[row] = inverse_B[row][col];
      }

      for (int row = 0; row < m; ++row) {
        if (row != leaving_index) {
          inverse_B[row][col] =
              tmp_column[row] + eta[row] * tmp_column[leaving_index];
        } else {
          inverse_B[row][col] = eta[row] * tmp_column[leaving_index];
        }
      }
    }

    basic_coefficients[leaving_index] = c[entering_index];
    basic_solutions[leaving_index] = minimal_ratio_test;
    for (int i = 0; i < m; ++i) {
      if (i != leaving_index) {
        basic_solutions[i] -= minimal_ratio_test * exchange_reduction[i];
      }
    }
    objective_value += minimal_ratio_test * best_entering_cost;

    // DEBUGGING
    // DLOG << "ratio_test " << minimal_ratio_test << '\n';
    // for (int i = 0; i < n; ++i) {
    //   std::cout << c[i] << ' ';
    // }
    // std::cout << '\n';
    // for (int row = 0; row < m; ++row) {
    //   for (int col = 0; col < m; ++col) {
    //     if (col != m - 1)
    //       std::cout << inverse_B[row][col] << ' ';
    //     else
    //       std::cout << inverse_B[row][col] << '\n';
    //   }
    // }
    DLOG_EVERY_N(INFO, 100) << "Iteration "
                            << iteration_pos
                            // << ", entering_index is: " << entering_index
                            << ", objective_value: " << objective_value;
  }  // End of Simplex loop
  // The solver couldn't solve it under n iterations.
  return {INIT, basic_solutions, basic_indices};
}
}  // namespace compute_tools
