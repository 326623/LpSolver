#include <iostream>
#include <fstream>

#include <gflags/gflags.h>

#include "jlp/linear_solver.h"

DEFINE_string(input_file, "",
              "The directory to the input of the file, which has the format as "
              "M N <vector c> <vector b> <matrix A>");

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

  for (int i = 0; i < n; ++i) {
    input_problem >> c[i];
    c[i] = -c[i];
  }

  for (int i = 0; i < m; ++i)
    input_problem >> b[i];

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      input_problem >> A[i][j];

  // call solve
  std::cout << solve(A, b, c) << '\n';
  return 0;
}
