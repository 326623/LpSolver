/* Copyright (C) 2018 New Joy - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3
 *
 *
 * You should have received a copy of the GPLv3 license with
 * this file. If not, please visit https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: yangqp5@outlook.com (New Joy)
 *
 */

#ifndef _JLP_LINEAR_SOLVER_H_
#define _JLP_LINEAR_SOLVER_H_
#include <vector>
#include <limits>

namespace compute_tools {
// Forget about generic code, it is so much harder to implement. Go easy on
// yourself
// template <template <typename> typename Matrix,
//           template <typename> typename Vector,
//           typename T>
// void solve(Matrix<T> A, Vector<T> b, Vector<T> c) {
//   int
//   Matrix<T> B()
// }

// A is a matrix of size m x n. b is a vector of size m, c is a vector of size
// n.
void solve(const std::vector<std::vector<double>>& A,
           const std::vector<double>& b, std::vector<double>& c,
           int num_iterations = 10) {
  int m = A.size();
  int n = A.front().size();
  assert(m > 0 && n > 0 && n >= m && "m == 0 or n == 0 or n < m");

  // Initialization
  std::vector<std::vector<double>> inverse_B{m, std::vector(m, 0)};
  for (int i = 0; i < m; ++i)
    inverse_B[i][i] = 1.0;

  std::vector<int> basic_indices{m};
  std::vector<int> nonbasic_indices{n - m};
  std::vector<double> coefficients{m};
  std::vector<double> basic_solutions{m};
  for (int offset = n - m, int i = offset; i < n; ++i) {
    coefficients[i - offset] = c[i];
    basic_indices[i - offset] = i;
    basic_solutions[i - offset] = b[i - offset];
  }

  for (int i = 0; i < n - m; ++i)
    nonbasic_indices[i] = i;

  std::vector<double> simplex_multiplier{m};
  std::vector<double> exchange_reduction{m};
  std::vector<double> eta{m};

  for (int iteration_pos = 0; iteration_pos < num_iterations; ++iteration_pos) {
    // Compute simplex multiplier
    for (int i = 0; i < m; ++i) {
      simplex_multiplier[i] = 0.0;
      for (int j = 0; j < m; ++j) {
        simplex_multiplier[i] += coefficients[j] * inverse_B[j][i];
      }
    }

    // Pricing, choose the what gives the lowest
    int entering_index = -1;
    // Any positive value that is the lowest means it is optimal
    double best_entering_cost = 1.0;
    for (int index : nonbasic_indices) {
      double entering_cost = c[index];
      for (int i = 0; i < m; ++i) {
        entering_cost -= simplex_multiplier[i] * A[i][index];
      }
      if (entering_cost < best_entering_cost) {
        best_entering_cost = entering_cost;
        entering_index = index;
      }
    }

    // WARNING: Please be careful with floating point issues
    if (best_entering_cost >= 0 || entering_index == -1) {
      // Optimal
      return;
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
      }
      if (ratio_test < minimal_ratio_test) {
        leaving_index = i;
        minimal_ratio_test = ratio_test;
      }
    }

    if (leaving_index == -1) {
      // Unbounded
      return;
    }

    // Need to compute a column \eta of the elementary matrix that is used to
    // update the inverse of B for the next iteration
    double alpha_leaving = exchange_reduction[leaving_index];
    for (int i = 0; i < m; ++i) {
      eta[i] = - exchange_reduction[i] / alpha_leaving;
    }
    eta[leaving_index] = 1 / alpha_leaving;

    // Update
    basic_indices[leaving_index] = entering_index;

    coefficients[leaving_index] = 
  }
}
} // namespace compute_tools

#endif /* _JLP_LINEAR_SOLVER_H_ */
