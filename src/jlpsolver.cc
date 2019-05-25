#include <iostream>
#include <fstream>
#include <tuple>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "jlp/linear_solver.h"

DEFINE_string(input_file, "",
              "The directory to the input of the file, which has the format as "
              "M N <vector c> <vector b> <matrix A>");

// temporary code snippet
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp = 1)
    // as in how many unit of precision, let's default to 1
{
  const auto diff = std::abs(x - y);
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return diff <= std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         // unless the result is subnormal
         // https://en.wikipedia.org/wiki/Denormal_number
         diff < std::numeric_limits<T>::min();
}

static bool ValidateInputFile(const char* flagname, const std::string& filename) {
  if (filename.size() == 0) {
    std::cerr << "Please specify a file name for: " << flagname << std::endl;
    return false;
  } else {
    std::ifstream test_open(filename);
    if (!test_open.good()) {
      std::cerr << "Could not find any file by the name: " << filename << std::endl;
      return false;
    }
  }
  return true;
}

// Check if one specify the correct file for input
DEFINE_validator(input_file, &ValidateInputFile);

int main(int argc, char* argv[]) {
  using namespace compute_tools;

  // True means remove flags, only argument left, "-i nope" the "-i" will be
  // removed.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  int m, n;
  // Reads in the problem
  std::ifstream input_problem(FLAGS_input_file);
  input_problem >> m >> n;
  std::vector<std::vector<double>> A(m, std::vector<double>(n));
  std::vector<double> b(m), c(n);

  LOG(WARNING) << "Current implementation would change the sign of the cost "
                  "coefficient vector c to stay in consistent with the our "
                  "baseline implementations.\n";
  LOG(WARNING) << "Current implementation change the sign of the objective as "
                  "well\n";

  for (int i = 0; i < n; ++i) {
    input_problem >> c[i];
    c[i] = -c[i];
  }

  for (int i = 0; i < m; ++i)
    input_problem >> b[i];

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      input_problem >> A[i][j];

  ProblemStatus status;
  double objective_value = 0.0;
  int num_iteration = 1000;
  std::vector<double> basic_solution;
  std::vector<int> basic_indices;
  std::tie(status, basic_solution, basic_indices) = Solve(A, b, c, num_iteration);
  DCHECK_EQ(basic_solution.size(), basic_indices.size()) << "size unmatched";
  // Checks feasibility
  for (int row = 0; row < m; ++row) {
    double left_b = 0.0;
    // all nonbasic indices' value is zero
    for (int i = 0; i < static_cast<int>(basic_indices.size()); ++i) {
      left_b += A[row][basic_indices[i]] * basic_solution[i];
    }
    DCHECK(almost_equal(left_b, b[row], 10)) << "solution is infeasible.";
  }

  for (int i = 0; i < m; ++ i) {
    std::cout << "x_" << basic_indices[i] << " = " << basic_solution[i]
              << '\n';
  }
  for (int i = 0; i < static_cast<int>(basic_indices.size()); ++i) {
    objective_value += c[basic_indices[i]] * basic_solution[i];
  }
  // call solve
  if (status == ProblemStatus::INIT) {
    std::cout << "Optimal not reached for " << num_iteration << " of iterations\n";
    std::cout << "Current Objective value is: " << -objective_value << '\n';
  }
  std::cout << -objective_value << '\n';
  return 0;
}
