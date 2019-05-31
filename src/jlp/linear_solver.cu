#include <tuple>
#include <vector>

#include "jlp/linear_solver.cuh"
#include "jlp/status.h"

std::tuple<ProblemStatus, std::vector<float>, std::vector<int>> CudaSolve(
    const std::vector<std::vector<float>>& host_A,
    const std::vector<float>& host_b, std::vector<float>& host_c,
    int num_iterations = 1000) {
  int m = host_A.size();
  DCHECK(m > 0) << "m == 0";
  int n = host_A.front().size();
  DCHECK(n > 0 && n >= m) << "m == 0 or n == 0 or n < m";
  DCHECK(n == host_c.size()) << "Incompatibility of size between A and c";
  DCHECK(m == host_b.size()) << "Incompatibility of size between A and b";

  // In this implementation, try less data transfer between CPU and GPU
  float* device_A;
  float* device_b;
  float* device_c;
  cudaMalloc(&device_A, (m * n) * sizeof(float));
  cudaMalloc(&device_b, m * sizeof(float));
  cudaMalloc(&device_c, n * sizeof(float));
  cudaMemcpy(device_A, host_A.data(), (m * n) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b.data(), m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_c, host_c.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  float* device_inv_B;
  int* device_basic_indices;
  int* device_nonbasic_indices;
  float* device_basic_coefficients;
  float* device_basic_solutions;
  cudaMalloc(&device_inv_B, (m * m) * sizeof(float));
  cudaMalloc(&device_basic_indices, m * sizeof(int));
  cudaMalloc(&device_nonbasic_indices, (n - m) * sizeof(int));
  cudaMalloc(&device_basic_coefficients, m * sizeof(float));
  cudaMalloc(&device_basic_solutions, m * sizeof(float));

  // Initialize these vectors with kernel
  InitializeVar<<<64, 64>>>(device_b, device_c, device_inv_B,
                            device_basic_coefficients, device_basic_indices,
                            device_nonbasic_indices, device_basic_solutions, m,
                            n);

  // Intermediate variables to aid computation
  float* device_simplex_multiplier;
  float* device_exchange_reduction;
  // For the update of inverse B
  float* device_tmp_column;
  float* device_eta;
  // initial objective value
  float* device_objective_value;
  cudaMalloc(&device_simplex_multiplier, m * sizeof(float));
  cudaMalloc(&device_exchange_reduction, m * sizeof(float));
  cudaMalloc(&device_tmp_column, m * sizeof(float));
  cudaMalloc(&device_eta, m * sizeof(float));
  cudaMalloc(&device_objective_value, sizeof(float));

  int* device_entering_index;
  int* device_nonbasic_picked;
  float* device_best_entering_cost;
  cudaMalloc(&device_entering_index, sizeof(int));
  cudaMalloc(&device_nonbasic_picked, sizeof(int));
  cudaMalloc(&device_best_entering_cost, sizeof(float));

  for (int iteration_pos = 0; iteration_pos < num_iterations; ++iteration_pos) {
    ComputeSimplexMultiplier<<<64, 64>>>(
        device_simplex_multiplier, device_basic_coefficients, device_inv_B, m);

    // __device__ int entering_index = -1;
    // __device__ int nonbasic_picked = -1;
    // __device__ float best_entering_cost = 1.0;
  }

  // Freeing cuda memory
  cudaFree(device_A);
  cudaFree(device_b);
  cudaFree(device_c);

  cudaFree(device_inv_B);
  cudaFree(device_basic_indices);
  cudaFree(device_nonbasic_indices);
  cudaFree(device_basic_coefficients);
  cudaFree(device_basic_solutions);

  cudaFree(device_simplex_multiplier);
  cudaFree(device_exchange_reduction);
  cudaFree(device_tmp_column);
  cudaFree(device_eta);
  cudaFree(device_objective_value);

  cudaFree(device_entering_index);
  cudaFree(device_nonbasic_picked);
  cudaFree(device_best_entering_cost);

  // return std::tuple<ProblemStatus, std::vector<float>, std::vector<int>>(
  //     {ProblemStatus::INIT, std::vector<float>(1), std::vector<int>(1)});
  std::vector<float> a(1);
  std::vector<int> b(1);
  return {INIT, {}, {}};
}

}